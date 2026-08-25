"""
    Copyright (c) Microsoft Corporation. All rights reserved.
    Licensed under the MIT License.

    Provenance verification for HDF5 (.h5) Keras model files.

    ``tensorflow.keras.models.load_model`` on the legacy HDF5 format
    deserializes the model configuration, and Keras ``Lambda`` layers
    persist their Python callable as marshalled bytecode that is
    reconstructed with ``marshal.loads()`` and executed on load. A crafted
    ``.h5`` file can therefore result in arbitrary code execution
    (CWE-502) the moment it is loaded, regardless of ``compile=False``.

    This module enforces a fail-closed trust policy before any HDF5 model
    is handed to Keras for deserialization:

    1. The model file must resolve to an exact path that is registered in
       the repository-controlled trust manifest (``trusted_models.json``).
       Matching is done on the fully resolved path -- not on filename --
       so an attacker-supplied file cannot inherit trust simply by being
       named ``custom_classifier.h5``.
    2. The file's SHA-256 digest must match the digest recorded in the
       manifest for that exact path.
    3. Symlinks are rejected outright, and the bytes that are hashed are
       the exact same bytes that are subsequently handed to Keras (via an
       in-memory ``h5py.File``), which removes the TOCTOU window between
       "hash the file" and "load the file".
    4. The (still-untrusted) model configuration embedded in the HDF5
       file is inspected *before* it is given to Keras, and any ``Lambda``
       layer or layer type outside of a small, explicit allow-list is
       rejected. This is defense in depth in case a trusted-looking file
       is ever replaced/corrupted, and because TensorFlow 2.4.1 predates
       any ``safe_mode`` protections in Keras.
"""

from __future__ import annotations

import errno
import hashlib
import io
import json
import os
import re
import stat
from typing import Any, Dict, Optional, Tuple

# Repository root: this file lives at "<repo_root>/src/model_security.py".
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Default location of the trust manifest, tracked in the repository.
DEFAULT_MANIFEST_PATH = os.path.join(_REPO_ROOT, "src", "trusted_models.json")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# Layer / initializer / regularizer class names that are known-safe and are
# actually used by the models shipped with this repository (MobileNetV2
# based encoder + small classification heads), plus a handful of other
# common, non-executable Keras primitives. Anything not on this list is
# rejected, and "Lambda" (and its functional-API sibling "TFOpLambda") is
# always rejected even if it were added here by mistake.
ALLOWED_LAYER_CLASS_NAMES = frozenset(
    {
        # Containers / graph structure
        "Functional",
        "Sequential",
        "Model",
        "InputLayer",
        "Network",
        # Core / common layers
        "Dense",
        "Dropout",
        "Activation",
        "Flatten",
        "Reshape",
        "Permute",
        "RepeatVector",
        "Concatenate",
        "Add",
        "Subtract",
        "Multiply",
        "Average",
        "Maximum",
        "Minimum",
        "Softmax",
        "ReLU",
        "LeakyReLU",
        "PReLU",
        "ELU",
        "ThresholdedReLU",
        # Convolutional / pooling layers used by MobileNetV2
        "Conv2D",
        "DepthwiseConv2D",
        "SeparableConv2D",
        "ZeroPadding2D",
        "GlobalAveragePooling2D",
        "GlobalMaxPooling2D",
        "MaxPooling2D",
        "AveragePooling2D",
        "UpSampling2D",
        "Cropping2D",
        # Normalization
        "BatchNormalization",
        "LayerNormalization",
        # Initializers / regularizers / constraints referenced from configs
        "Zeros",
        "Ones",
        "Constant",
        "RandomNormal",
        "RandomUniform",
        "TruncatedNormal",
        "GlorotUniform",
        "GlorotNormal",
        "HeUniform",
        "HeNormal",
        "L1L2",
    }
)

# Layer class names that are always rejected, even if erroneously present
# in the allow-list above, because they can execute arbitrary code embedded
# in the model file.
DENYLISTED_LAYER_CLASS_NAMES = frozenset({"Lambda", "TFOpLambda"})


class ModelTrustError(Exception):
    """Raised when an HDF5 model file fails provenance/trust verification."""


def _load_manifest(manifest_path: str) -> Dict[str, Dict[str, str]]:
    """Loads and validates the trusted-model manifest.

    Args:
        manifest_path (str): Path to the JSON manifest file.

    Returns:
        Dict[str, Dict[str, str]]: Mapping of repo-root-relative POSIX
            paths to a dict containing at least a "sha256" key.

    Raises:
        ModelTrustError: If the manifest is missing, unreadable, not valid
            JSON, or malformed.
    """
    if not os.path.isfile(manifest_path):
        raise ModelTrustError(
            f"Trusted model manifest not found at '{manifest_path}'. "
            "Refusing to load any HDF5 model without a manifest."
        )
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelTrustError(
            f"Could not read/parse trusted model manifest '{manifest_path}': {exc}"
        ) from exc

    if not isinstance(raw, dict):
        raise ModelTrustError(
            f"Malformed trusted model manifest '{manifest_path}': "
            "expected a JSON object mapping paths to digest info."
        )

    manifest: Dict[str, Dict[str, str]] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            raise ModelTrustError(
                f"Malformed trusted model manifest entry for {key!r}."
            )
        digest = value.get("sha256")
        if not isinstance(digest, str) or not _SHA256_RE.match(digest.lower()):
            raise ModelTrustError(
                f"Malformed or missing sha256 digest for manifest entry {key!r}."
            )
        manifest[key] = {"sha256": digest.lower()}

    return manifest


def _relative_manifest_key(real_path: str) -> str:
    """Converts an absolute, fully-resolved path to a POSIX-style path
    relative to the repository root, suitable for manifest lookups.
    """
    rel = os.path.relpath(real_path, _REPO_ROOT)
    return rel.replace(os.sep, "/")


def _read_bytes_no_symlink(model_path: str) -> bytes:
    """Opens ``model_path`` refusing to follow a symlink at the final
    path component, verifies it is a regular file, and returns its bytes.

    Reading the bytes into memory (rather than handing a path to Keras)
    means the digest we verify is exactly what gets deserialized later,
    closing the TOCTOU window between validation and load.

    Args:
        model_path (str): User-supplied path to the candidate model file.

    Returns:
        bytes: The full contents of the file.

    Raises:
        ModelTrustError: If the path does not exist, is a symlink, is not
            a regular file, or cannot be read.
    """
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    try:
        fd = os.open(model_path, flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise ModelTrustError(
                f"Refusing to load '{model_path}': path is a symlink. "
                "Symlinked model files are not trusted."
            ) from exc
        raise ModelTrustError(
            f"Could not open model file '{model_path}': {exc}"
        ) from exc

    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ModelTrustError(
                f"Refusing to load '{model_path}': not a regular file."
            )
        with os.fdopen(fd, "rb") as fh:
            fd = -1  # ownership transferred to the file object
            return fh.read()
    finally:
        if fd >= 0:
            os.close(fd)


def verify_and_read_trusted_model(
    model_path: str,
    manifest_path: Optional[str] = None,
) -> Tuple[bytes, str]:
    """Validates the provenance of an HDF5 model file against a trusted
    manifest and returns its verified bytes.

    Args:
        model_path (str): User-supplied path to the ``.h5`` model.
        manifest_path (Optional[str]): Path to the trust manifest. Defaults
            to the repository-tracked manifest.

    Returns:
        Tuple[bytes, str]: The verified file bytes, and the manifest key
            (repo-relative path) it matched.

    Raises:
        ModelTrustError: On any provenance/trust failure. Fails closed.
    """
    if not model_path or not isinstance(model_path, str):
        raise ModelTrustError("A model path must be provided.")

    if os.path.islink(model_path):
        raise ModelTrustError(
            f"Refusing to load '{model_path}': path is a symlink. "
            "Symlinked model files are not trusted."
        )

    if not os.path.exists(model_path):
        raise ModelTrustError(f"Model file not found: '{model_path}'.")

    manifest = _load_manifest(manifest_path or DEFAULT_MANIFEST_PATH)

    real_path = os.path.realpath(model_path)
    manifest_key = _relative_manifest_key(real_path)

    entry = manifest.get(manifest_key)
    if entry is None:
        raise ModelTrustError(
            f"'{model_path}' is not a recognized trusted model. Only model "
            "files registered by exact path/digest in "
            f"'{manifest_path or DEFAULT_MANIFEST_PATH}' may be loaded. "
            "See README.md for how to register a trusted model."
        )

    data = _read_bytes_no_symlink(model_path)

    actual_digest = hashlib.sha256(data).hexdigest()
    expected_digest = entry["sha256"]
    if actual_digest != expected_digest:
        raise ModelTrustError(
            f"Refusing to load '{model_path}': SHA-256 digest mismatch "
            f"(expected {expected_digest}, got {actual_digest}). The file "
            "may have been tampered with or is out of date."
        )

    return data, manifest_key


def _walk_for_layer_class_names(node: Any, found: set) -> None:
    """Recursively collects every ``class_name`` value found in a parsed
    Keras model configuration, including nested Functional/Sequential
    sub-models.
    """
    if isinstance(node, dict):
        class_name = node.get("class_name")
        if isinstance(class_name, str):
            found.add(class_name)
        for value in node.values():
            _walk_for_layer_class_names(value, found)
    elif isinstance(node, list):
        for item in node:
            _walk_for_layer_class_names(item, found)


def assert_no_unsafe_layers(model_bytes: bytes) -> None:
    """Inspects an HDF5 Keras model's embedded configuration and rejects
    it if it contains ``Lambda``/``TFOpLambda`` layers or any layer class
    name outside of the explicit allow-list.

    This is defense in depth performed *before* the bytes are handed to
    Keras for deserialization, since TensorFlow 2.4.1 has no ``safe_mode``
    protection against Lambda-layer bytecode execution.

    Args:
        model_bytes (bytes): Verified bytes of the HDF5 model file.

    Raises:
        ModelTrustError: If the model configuration cannot be parsed, or
            contains a denylisted or unrecognized layer/class type.
    """
    import h5py  # Local import: only needed for models we intend to load.

    try:
        with h5py.File(io.BytesIO(model_bytes), mode="r") as h5file:
            raw_config = h5file.attrs.get("model_config")
    except Exception as exc:  # noqa: BLE001 - surface as a trust failure
        raise ModelTrustError(
            f"Could not parse HDF5 model structure for validation: {exc}"
        ) from exc

    if raw_config is None:
        raise ModelTrustError(
            "HDF5 file has no 'model_config' attribute; cannot validate "
            "model structure prior to loading."
        )

    if isinstance(raw_config, bytes):
        raw_config = raw_config.decode("utf-8")

    try:
        config = json.loads(raw_config)
    except json.JSONDecodeError as exc:
        raise ModelTrustError(
            f"Could not parse model_config JSON for validation: {exc}"
        ) from exc

    class_names: set = set()
    _walk_for_layer_class_names(config, class_names)

    denylisted = class_names & DENYLISTED_LAYER_CLASS_NAMES
    if denylisted:
        raise ModelTrustError(
            "Refusing to load model: contains disallowed executable-code "
            f"layer type(s) {sorted(denylisted)} (e.g. Lambda layers can "
            "embed and execute arbitrary Python bytecode)."
        )

    unknown = class_names - ALLOWED_LAYER_CLASS_NAMES
    if unknown:
        raise ModelTrustError(
            "Refusing to load model: contains unapproved/custom layer or "
            f"object type(s) {sorted(unknown)} that are not on the "
            "reviewed allow-list."
        )


def load_verified_keras_model(
    model_path: str,
    manifest_path: Optional[str] = None,
    compile: bool = False,  # noqa: A002 - mirrors keras.models.load_model kw
):
    """Verifies provenance/trust of an HDF5 Keras model file and loads it.

    The file is never handed to Keras by path: the exact bytes that were
    hashed and structurally validated are the bytes that get deserialized,
    which removes the TOCTOU window between verification and loading.

    Args:
        model_path (str): User-supplied path to the ``.h5`` model file.
        manifest_path (Optional[str]): Path to the trust manifest. Defaults
            to the repository-tracked manifest.
        compile (bool): Forwarded to ``tensorflow.keras.models.load_model``.
            Defaults to False since compilation is not required to run
            inference/feature-extraction.

    Returns:
        tensorflow.keras.Model: The loaded, verified model.

    Raises:
        ModelTrustError: If provenance/trust verification fails. In this
            case ``tensorflow.keras.models.load_model`` is never called.
    """
    data, _manifest_key = verify_and_read_trusted_model(model_path, manifest_path)
    assert_no_unsafe_layers(data)

    # Imported lazily so that pure validation logic (and its tests) does
    # not require TensorFlow to be installed.
    import h5py
    from tensorflow.keras.models import load_model

    with h5py.File(io.BytesIO(data), mode="r") as h5file:
        return load_model(h5file, compile=compile)
