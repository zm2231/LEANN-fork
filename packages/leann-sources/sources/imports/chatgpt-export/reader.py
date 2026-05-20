"""ChatGPT export source-registry reader."""

from __future__ import annotations

from collections.abc import Iterator

from leann_sources.base import Chunk
from leann_sources.manifest import SourceManifest
from leann_sources.readers.documents import DocumentSourceReader


class ChatGPTExportReader(DocumentSourceReader):
    def __init__(self, manifest: SourceManifest):
        super().__init__(manifest)

    def load_documents(self, *, max_count: int = -1):
        from apps.chatgpt_data.chatgpt_reader import ChatGPTReader

        return ChatGPTReader(concatenate_conversations=True).load_data(
            input_dir=str(self.path),
            max_count=max_count,
        )

    def iter_chunks(self, since=None) -> Iterator[Chunk]:
        for chunk in super().iter_chunks(since=since):
            event_time = chunk.metadata.get("event_time")
            if event_time and "modified_at" not in chunk.metadata:
                chunk.metadata["modified_at"] = event_time
                chunk.metadata["modified_at_synthesized"] = True
            yield chunk
