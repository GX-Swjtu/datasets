import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import boto3
import pytest


@pytest.fixture
def storage(monkeypatch):
    # The adapter only needs settings.S3; do not boot ingestion, databases or models.
    config = {
        "access_key": "fixture", "secret_key": "fixture-secret", "bucket": "physical-bucket",
        "prefix_path": "documents", "endpoint_url": "https://external-s3.example.test",
        "region_name": "provider-region", "signature_version": "v4", "addressing_style": "path",
    }
    common = ModuleType("common")
    common.settings = SimpleNamespace(S3=config)
    decorators = ModuleType("common.decorator")
    decorators.singleton = lambda cls: cls
    monkeypatch.setitem(sys.modules, "common", common)
    monkeypatch.setitem(sys.modules, "common.decorator", decorators)
    client = Mock()
    factory = Mock(return_value=client)
    monkeypatch.setattr(boto3, "client", factory)
    spec = importlib.util.spec_from_file_location("s3_stream_test_adapter", Path(__file__).parents[4] / "rag/utils/s3_conn.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RAGFlowS3(), client, factory


def test_external_endpoint_region_and_addressing_are_reused(storage):
    _, _, factory = storage
    args = factory.call_args.kwargs
    assert args["endpoint_url"] == "https://external-s3.example.test"
    assert args["region_name"] == "provider-region"
    assert args["config"].signature_version == "v4"
    assert args["config"].s3["addressing_style"] == "path"


def test_stat_preserves_physical_bucket_and_logical_prefix(storage):
    adapter, client, _ = storage
    client.head_object.return_value = {"ContentLength": 300000000}
    assert adapter.stat("dataset-id", "video-id") == {"ContentLength": 300000000}
    client.head_object.assert_called_once_with(Bucket="physical-bucket", Key="documents/dataset-id/video-id")
    client.get_object.assert_not_called()


@pytest.mark.parametrize("byte_range", [None, (0, 1), (200000000, 299999999)])
def test_get_stream_uses_standard_s3_range_without_reading_body(storage, byte_range):
    adapter, client, _ = storage
    body = Mock()
    client.get_object.return_value = {"Body": body}
    assert adapter.get_stream("dataset-id", "video-id", byte_range=byte_range, etag='"revision"') == {"Body": body}
    expected = {"Bucket": "physical-bucket", "Key": "documents/dataset-id/video-id", "IfMatch": '"revision"'}
    if byte_range is not None:
        expected["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"
    client.get_object.assert_called_once_with(**expected)
    body.read.assert_not_called()
    body.close.assert_not_called()
