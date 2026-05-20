"""Shared manifest field-mapping helpers for source readers."""

from __future__ import annotations

import ast
import re
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from leann_sources import transforms
from leann_sources.base import Chunk
from leann_sources.manifest import SourceManifest


class MissingRequiredFieldError(ValueError):
    pass


class SafeRecord(dict):
    def __missing__(self, key: str) -> str:
        return ""


class ManifestMappingMixin:
    manifest: SourceManifest

    def _chunk_from_record(
        self,
        record: Mapping[str, Any],
        *,
        row_index: int = 0,
        text: str | None = None,
        document_counts: defaultdict[Any, int] | None = None,
    ) -> Chunk:
        rendered_text = text if text is not None else self._render_text(record)
        metadata: dict[str, Any] = {}
        if document_counts is None:
            document_counts = defaultdict(int)

        for field_name, mapping in self.manifest.fields.items():
            value = self._value_from_mapping(
                mapping,
                record,
                row_index=row_index,
                text=rendered_text,
                document_counts=document_counts,
            )
            if value is None:
                if mapping.get("required"):
                    raise MissingRequiredFieldError(f"missing required field '{field_name}'")
                continue
            metadata[field_name] = value
            if mapping.get("synthesized_from"):
                metadata[f"{field_name}_synthesized"] = True

        return Chunk(text=rendered_text, metadata=metadata)

    def _value_from_mapping(
        self,
        mapping: Mapping[str, Any],
        record: Mapping[str, Any],
        *,
        row_index: int,
        text: str,
        document_counts: defaultdict[Any, int],
    ) -> Any:
        if "value" in mapping:
            value = mapping["value"]
        elif "auto" in mapping:
            value = self._auto_value(mapping["auto"], record, row_index, document_counts)
        elif mapping.get("extraction") == "regex":
            source_text = text
            if "source" in mapping:
                source_value = self._source_value(record, mapping["source"])
                source_text = "" if source_value is None else str(source_value)
            return transforms.regex_extract(source_text, mapping["pattern"])
        else:
            value = self._source_value(record, mapping["source"])

        transform = mapping.get("transform")
        if transform and value is not None:
            value = self._apply_transform(transform, value)
        return value

    def _auto_value(
        self,
        auto: str,
        record: Mapping[str, Any],
        row_index: int,
        document_counts: defaultdict[Any, int],
    ) -> Any:
        if auto == "row_index":
            return row_index
        if auto == "row_index_within_source_document_id":
            document_id = record.get("source_document_id") or record.get("path") or "__default__"
            value = document_counts[document_id]
            document_counts[document_id] += 1
            return value
        raise ValueError(f"unknown auto mapping '{auto}'")

    def _source_value(self, record: Mapping[str, Any], source: str) -> Any:
        if source in record:
            return record[source]
        if "." in source:
            tail = source.rsplit(".", 1)[1]
            if tail in record:
                return record[tail]
        current: Any = record
        for part in source.split("."):
            if isinstance(current, Mapping) and part in current:
                current = current[part]
            else:
                return None
        return current

    def _apply_transform(self, transform: str, value: Any) -> Any:
        if "(" not in transform:
            return getattr(transforms, transform)(value)

        name, raw_args = transform.split("(", 1)
        raw_args = raw_args[:-1]
        if name == "format":
            template = ast.literal_eval(raw_args)
            return transforms.format(template, value)
        if name == "lookup":
            return value
        return getattr(transforms, name)(value)

    def _render_text(self, record: Mapping[str, Any]) -> str:
        template = self.manifest.chunking.get("text_template")
        if template:
            return template.format_map(SafeRecord(record))
        for key in ("text", "content", "body", "message", "title"):
            if key in record and record[key] is not None:
                return str(record[key])
        return " ".join(str(value) for value in record.values() if value is not None)


def extract_urls(text: str) -> list[str]:
    return re.findall(r"https?://[^\s>]+", text)
