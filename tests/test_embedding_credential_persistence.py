"""Credential material must remain runtime-only during embedding."""

from leann.api import _persistable_embedding_options


def test_persistable_embedding_options_remove_credentials():
    options = {
        "base_url": "https://embeddings.example.test/v1",
        "api_key": "must-not-persist",
        "query_prompt_template": "query: {text}",
    }

    persisted = _persistable_embedding_options(options)

    assert persisted == {
        "base_url": "https://embeddings.example.test/v1",
        "query_prompt_template": "query: {text}",
    }
