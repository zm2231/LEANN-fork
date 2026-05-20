"""Manifest loading and validation for LEANN sources."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

TEMPORAL_AXES = {"created_at", "modified_at", "event_time", "indexed_at"}

FIELD_NAMES = {
    "created_at",
    "modified_at",
    "event_time",
    "event_time_local",
    "author",
    "activity_type",
    "participant_ids",
    "source_type",
    "source_id",
    "source_url",
    "project_id",
    "parent_ref",
    "mentioned_urls",
    "mentioned_refs",
    "mentioned_files",
    "source_document_id",
    "chunk_seq",
    "event_id",
    "event_date",
    "speaker",
    "public_citation_allowed",
    "review_required",
}

TRANSFORM_NAMES = {
    "unix_to_utc_iso",
    "core_data_epoch_to_utc_iso",
    "cocoa_ns_to_utc_iso",
    "webkit_epoch_to_utc_iso",
    "regex_extract",
    "format",
    "lookup",
    "list_of_handles",
}

MANIFEST_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "name",
        "category",
        "display_name",
        "version",
        "manifest_version",
        "data",
        "fields",
        "chunking",
        "privacy",
    ],
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "category": {"type": "string", "minLength": 1},
        "display_name": {"type": "string", "minLength": 1},
        "version": {"type": "string", "minLength": 1},
        "manifest_version": {"type": "string", "const": "1.0"},
        "data": {
            "type": "object",
            "additionalProperties": True,
            "required": ["type"],
            "properties": {
                "type": {
                    "type": "string",
                    "enum": [
                        "sqlite",
                        "filesystem",
                        "api",
                        "export_zip",
                        "cloud_drive",
                        "live_stream",
                    ],
                },
                "default_path": {"type": "string"},
                "permissions": {"type": "array", "items": {"type": "string"}},
            },
        },
        "auth": {
            "type": "object",
            "additionalProperties": True,
            "properties": {
                "type": {
                    "type": "string",
                    "enum": ["none", "env_var", "keychain", "oauth", "api_key"],
                }
            },
        },
        "fields": {
            "type": "object",
            "minProperties": 1,
            "additionalProperties": {"$ref": "#/$defs/field_mapping"},
        },
        "importance": {"type": "object", "additionalProperties": True},
        "chunking": {
            "type": "object",
            "additionalProperties": True,
            "required": ["granularity"],
            "properties": {
                "granularity": {"type": "string", "minLength": 1},
                "max_chunk_tokens": {"type": "integer", "minimum": 1},
                "overlap_tokens": {"type": "integer", "minimum": 0},
            },
        },
        "privacy": {
            "type": "object",
            "additionalProperties": True,
            "required": ["tier"],
            "properties": {"tier": {"type": "string", "enum": ["tier_1", "tier_2", "tier_3"]}},
        },
        "connectors": {"type": "object", "additionalProperties": True},
        "reader": {"type": "string"},
    },
    "$defs": {
        "field_mapping": {
            "type": "object",
            "additionalProperties": False,
            "anyOf": [
                {"required": ["source"]},
                {"required": ["value"]},
                {"required": ["auto"]},
                {"required": ["extraction"]},
            ],
            "properties": {
                "source": {"type": "string"},
                "value": {},
                "auto": {"type": "string"},
                "transform": {"type": "string"},
                "required": {"type": "boolean"},
                "synthesized_from": {"type": "string", "enum": sorted(TEMPORAL_AXES)},
                "extraction": {"type": "string", "enum": ["regex"]},
                "pattern": {"type": "string"},
            },
        }
    },
}


class ManifestValidationError(ValueError):
    pass


@dataclass(frozen=True)
class SourceManifest:
    name: str
    category: str
    display_name: str
    version: str
    manifest_version: str
    data: dict[str, Any]
    fields: dict[str, dict[str, Any]]
    chunking: dict[str, Any]
    privacy: dict[str, Any]
    auth: dict[str, Any] | None = None
    importance: dict[str, Any] | None = None
    connectors: dict[str, Any] | None = None
    reader: str | None = None
    path: Path | None = None
    raw: dict[str, Any] | None = None

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], *, path: Path | None = None, validate: bool = True
    ) -> SourceManifest:
        if validate:
            validate_manifest_dict(data)
        return cls(
            name=data["name"],
            category=data["category"],
            display_name=data["display_name"],
            version=data["version"],
            manifest_version=data["manifest_version"],
            data=dict(data["data"]),
            fields={key: dict(value) for key, value in data["fields"].items()},
            chunking=dict(data["chunking"]),
            privacy=dict(data["privacy"]),
            auth=dict(data["auth"]) if "auth" in data else None,
            importance=dict(data["importance"]) if "importance" in data else None,
            connectors=dict(data["connectors"]) if "connectors" in data else None,
            reader=data.get("reader"),
            path=path,
            raw=dict(data),
        )

    @classmethod
    def load(cls, path: str | Path) -> SourceManifest:
        return load_manifest(path)

    def to_dict(self) -> dict[str, Any]:
        if self.raw is not None:
            return dict(self.raw)
        data: dict[str, Any] = {
            "name": self.name,
            "category": self.category,
            "display_name": self.display_name,
            "version": self.version,
            "manifest_version": self.manifest_version,
            "data": self.data,
            "fields": self.fields,
            "chunking": self.chunking,
            "privacy": self.privacy,
        }
        for key in ("auth", "importance", "connectors", "reader"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        return data


def load_manifest(path: str | Path) -> SourceManifest:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ManifestValidationError(f"{manifest_path} must contain a YAML object")
    return SourceManifest.from_dict(data, path=manifest_path)


def validate_manifest_dict(data: dict[str, Any]) -> None:
    validator = Draft202012Validator(MANIFEST_SCHEMA)
    errors = sorted(validator.iter_errors(data), key=lambda error: list(error.path))
    if errors:
        raise ManifestValidationError(_format_schema_error(errors[0]))

    for field_name, mapping in data["fields"].items():
        if field_name not in FIELD_NAMES:
            raise ManifestValidationError(
                f"unknown SIGNALS field '{field_name}'; add it to docs/dev/SIGNALS.md first"
            )
        transform = mapping.get("transform")
        if transform:
            _validate_transform(transform)
        if mapping.get("extraction") == "regex" and not mapping.get("pattern"):
            raise ManifestValidationError(f"field '{field_name}' regex extraction requires pattern")


def _format_schema_error(error: ValidationError) -> str:
    path = ".".join(str(part) for part in error.path)
    prefix = f"{path}: " if path else ""
    return f"{prefix}{error.message}"


def _validate_transform(transform: str) -> None:
    name = transform.split("(", 1)[0]
    if name not in TRANSFORM_NAMES:
        raise ManifestValidationError(f"unknown transform '{transform}'")
    if "(" in transform and not transform.endswith(")"):
        raise ManifestValidationError(f"malformed transform '{transform}'")
    if name == "format" and "{value}" not in transform:
        raise ManifestValidationError("format transform must include '{value}'")
