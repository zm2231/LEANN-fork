"""WeChat export source-registry reader."""

from __future__ import annotations

from leann_sources.manifest import SourceManifest
from leann_sources.readers.documents import DocumentSourceReader


class WeChatSourceReader(DocumentSourceReader):
    def __init__(self, manifest: SourceManifest):
        super().__init__(manifest)

    def load_documents(self, *, max_count: int = -1):
        from apps.history_data.wechat_history import WeChatHistoryReader

        return WeChatHistoryReader().load_data(
            input_dir=str(self.path),
            wechat_export_dir=str(self.path),
            max_count=max_count,
            concatenate_messages=True,
        )
