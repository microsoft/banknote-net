"""
    Copyright (c) Microsoft Corporation. All rights reserved.
    Licensed under the MIT License.

    Focused tests for src/model_security.py, the HDF5 model provenance
    verification used before any tensorflow.keras.models.load_model call.

    TensorFlow is mocked out via sys.modules so these tests stay fast and
    do not require the pinned TensorFlow/h5py runtime to be installed.
"""

import hashlib
import io
import json
import os
import shutil
import sys
import types
from unittest import mock

import h5py
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRUSTED_CLASSIFIER = os.path.join(
    REPO_ROOT, "src", "trained_models", "custom_classifier.h5"
)
DEFAULT_MANIFEST = os.path.join(REPO_ROOT, "src", "trusted_models.json")


@pytest.fixture(autouse=True)
def fake_tensorflow(monkeypatch):
    """Registers a stub tensorflow.keras.models module so that
    ``from tensorflow.keras.models import load_model`` succeeds and can be
    asserted on, without requiring the real (legacy, py3.7-only) TF 2.4.1
    to be installed in the test environment.
    """
    load_model_mock = mock.Mock(name="load_model", return_value="LOADED_MODEL")

    tf_module = types.ModuleType("tensorflow")
    tf_keras_module = types.ModuleType("tensorflow.keras")
    tf_keras_models_module = types.ModuleType("tensorflow.keras.models")
    tf_keras_models_module.load_model = load_model_mock
    tf_keras_module.models = tf_keras_models_module
    tf_module.keras = tf_keras_module

    monkeypatch.setitem(sys.modules, "tensorflow", tf_module)
    monkeypatch.setitem(sys.modules, "tensorflow.keras", tf_keras_module)
    monkeypatch.setitem(sys.modules, "tensorflow.keras.models", tf_keras_models_module)

    return load_model_mock


@pytest.fixture
def ms(monkeypatch):
    """Imports model_security fresh for each test."""
    sys.modules.pop("model_security", None)
    import model_security as module

    return module


def _write_manifest(path, entries):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(entries, fh)


def _fake_h5_bytes(layer_class_names):
    """Builds a minimal in-memory HDF5 file with a model_config attribute
    whose layers use the given class names.
    """
    buf = io.BytesIO()
    with h5py.File(buf, "w") as f:
        config = {
            "class_name": "Sequential",
            "config": {
                "layers": [{"class_name": name, "config": {}} for name in layer_class_names]
            },
        }
        f.attrs["model_config"] = json.dumps(config)
    return buf.getvalue()


# --------------------------------------------------------------------------
# Trusted model acceptance
# --------------------------------------------------------------------------


def test_trusted_model_is_loaded_via_verified_bytes(ms, fake_tensorflow):
    model = ms.load_verified_keras_model(TRUSTED_CLASSIFIER)

    assert model == "LOADED_MODEL"
    fake_tensorflow.assert_called_once()
    (h5_arg,), kwargs = fake_tensorflow.call_args
    assert isinstance(h5_arg, h5py.File)
    assert kwargs == {"compile": False}


def test_verify_and_read_trusted_model_returns_matching_digest(ms):
    data, key = ms.verify_and_read_trusted_model(TRUSTED_CLASSIFIER)

    assert key == "src/trained_models/custom_classifier.h5"
    with open(TRUSTED_CLASSIFIER, "rb") as fh:
        assert data == fh.read()


# --------------------------------------------------------------------------
# Untrusted / unknown path rejection
# --------------------------------------------------------------------------


def test_unknown_path_is_rejected_without_calling_load_model(ms, fake_tensorflow, tmp_path):
    untrusted = tmp_path / "not_registered.h5"
    untrusted.write_bytes(b"whatever bytes")

    with pytest.raises(ms.ModelTrustError, match="not a recognized trusted model"):
        ms.load_verified_keras_model(str(untrusted))

    fake_tensorflow.assert_not_called()


def test_same_filename_different_location_is_not_trusted_by_name_alone(
    ms, fake_tensorflow, tmp_path
):
    """A file sharing a trusted filename, but living outside the exact
    trusted path, must not inherit trust."""
    imposter = tmp_path / "custom_classifier.h5"
    shutil.copy(TRUSTED_CLASSIFIER, imposter)

    with pytest.raises(ms.ModelTrustError, match="not a recognized trusted model"):
        ms.load_verified_keras_model(str(imposter))

    fake_tensorflow.assert_not_called()


def test_nonexistent_path_is_rejected(ms, fake_tensorflow, tmp_path):
    missing = tmp_path / "does_not_exist.h5"

    with pytest.raises(ms.ModelTrustError, match="not found"):
        ms.load_verified_keras_model(str(missing))

    fake_tensorflow.assert_not_called()


# --------------------------------------------------------------------------
# Digest mismatch
# --------------------------------------------------------------------------


def test_digest_mismatch_is_rejected(ms, fake_tensorflow, tmp_path):
    tampered = tmp_path / "custom_classifier.h5"
    shutil.copy(TRUSTED_CLASSIFIER, tampered)
    with open(tampered, "r+b") as fh:
        fh.write(b"\x00\x00\x00\x00")

    manifest_path = tmp_path / "manifest.json"
    # Register the tampered file's *path* as trusted, but keep the digest
    # of the pristine repository copy, to isolate the digest check.
    with open(TRUSTED_CLASSIFIER, "rb") as fh:
        original_digest = hashlib.sha256(fh.read()).hexdigest()
    monkeypatch_key = os.path.relpath(str(tampered), ms._REPO_ROOT).replace(os.sep, "/")
    _write_manifest(manifest_path, {monkeypatch_key: {"sha256": original_digest}})

    with pytest.raises(ms.ModelTrustError, match="digest mismatch"):
        ms.load_verified_keras_model(str(tampered), manifest_path=str(manifest_path))

    fake_tensorflow.assert_not_called()


# --------------------------------------------------------------------------
# Malformed manifest handling
# --------------------------------------------------------------------------


def test_missing_manifest_file_fails_closed(ms, fake_tensorflow, tmp_path):
    missing_manifest = tmp_path / "does_not_exist.json"

    with pytest.raises(ms.ModelTrustError, match="manifest not found"):
        ms.load_verified_keras_model(
            TRUSTED_CLASSIFIER, manifest_path=str(missing_manifest)
        )

    fake_tensorflow.assert_not_called()


def test_invalid_json_manifest_fails_closed(ms, fake_tensorflow, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{not valid json")

    with pytest.raises(ms.ModelTrustError, match="Could not read/parse"):
        ms.load_verified_keras_model(
            TRUSTED_CLASSIFIER, manifest_path=str(manifest_path)
        )

    fake_tensorflow.assert_not_called()


def test_manifest_entry_missing_digest_fails_closed(ms, fake_tensorflow, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path, {"src/trained_models/custom_classifier.h5": {"not_sha256": "x"}}
    )

    with pytest.raises(ms.ModelTrustError, match="Malformed or missing sha256"):
        ms.load_verified_keras_model(
            TRUSTED_CLASSIFIER, manifest_path=str(manifest_path)
        )

    fake_tensorflow.assert_not_called()


def test_manifest_entry_with_malformed_digest_fails_closed(ms, fake_tensorflow, tmp_path):
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(
        manifest_path,
        {"src/trained_models/custom_classifier.h5": {"sha256": "not-a-hex-digest"}},
    )

    with pytest.raises(ms.ModelTrustError, match="Malformed or missing sha256"):
        ms.load_verified_keras_model(
            TRUSTED_CLASSIFIER, manifest_path=str(manifest_path)
        )

    fake_tensorflow.assert_not_called()


# --------------------------------------------------------------------------
# Symlink / path edge cases
# --------------------------------------------------------------------------


def test_symlink_to_trusted_model_is_rejected(ms, fake_tensorflow, tmp_path):
    link = tmp_path / "link.h5"
    link.symlink_to(TRUSTED_CLASSIFIER)

    with pytest.raises(ms.ModelTrustError, match="symlink"):
        ms.load_verified_keras_model(str(link))

    fake_tensorflow.assert_not_called()


def test_relative_path_to_trusted_model_still_resolves_and_loads(
    ms, fake_tensorflow, monkeypatch
):
    monkeypatch.chdir(REPO_ROOT)
    model = ms.load_verified_keras_model("./src/trained_models/custom_classifier.h5")

    assert model == "LOADED_MODEL"
    fake_tensorflow.assert_called_once()


# --------------------------------------------------------------------------
# Lambda-layer / unapproved custom object rejection
# --------------------------------------------------------------------------


def test_lambda_layer_config_is_rejected(ms, fake_tensorflow, tmp_path):
    data = _fake_h5_bytes(["InputLayer", "Dense", "Lambda"])
    model_path = tmp_path / "malicious.h5"
    model_path.write_bytes(data)

    manifest_path = tmp_path / "manifest.json"
    digest = hashlib.sha256(data).hexdigest()
    rel_key = os.path.relpath(str(model_path), ms._REPO_ROOT).replace(os.sep, "/")
    _write_manifest(manifest_path, {rel_key: {"sha256": digest}})

    with pytest.raises(ms.ModelTrustError, match="Lambda"):
        ms.load_verified_keras_model(str(model_path), manifest_path=str(manifest_path))

    fake_tensorflow.assert_not_called()


def test_unapproved_custom_layer_type_is_rejected(ms, fake_tensorflow, tmp_path):
    data = _fake_h5_bytes(["InputLayer", "TotallyUnreviewedCustomLayer"])
    model_path = tmp_path / "custom.h5"
    model_path.write_bytes(data)

    manifest_path = tmp_path / "manifest.json"
    digest = hashlib.sha256(data).hexdigest()
    rel_key = os.path.relpath(str(model_path), ms._REPO_ROOT).replace(os.sep, "/")
    _write_manifest(manifest_path, {rel_key: {"sha256": digest}})

    with pytest.raises(ms.ModelTrustError, match="unapproved/custom layer"):
        ms.load_verified_keras_model(str(model_path), manifest_path=str(manifest_path))

    fake_tensorflow.assert_not_called()


def test_real_trusted_models_contain_no_unsafe_layers(ms):
    for rel_path in (
        "src/trained_models/custom_classifier.h5",
        "src/trained_models/shallow_classifier.h5",
        "models/banknote_net_encoder.h5",
    ):
        data, _key = ms.verify_and_read_trusted_model(os.path.join(REPO_ROOT, rel_path))
        # Should not raise.
        ms.assert_no_unsafe_layers(data)


# --------------------------------------------------------------------------
# Manifest content sanity: repo-tracked models must be registered
# --------------------------------------------------------------------------


def test_trusted_manifest_covers_all_tracked_model_artifacts():
    with open(DEFAULT_MANIFEST, encoding="utf-8") as fh:
        manifest = json.load(fh)

    for rel_path in (
        "src/trained_models/custom_classifier.h5",
        "src/trained_models/shallow_classifier.h5",
        "models/banknote_net_encoder.h5",
    ):
        assert rel_path in manifest

        with open(os.path.join(REPO_ROOT, rel_path), "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        assert manifest[rel_path]["sha256"] == digest
