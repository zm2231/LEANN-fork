"""GitHub REST API source-registry reader."""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import datetime
from typing import Any

from leann_sources.base import ValidationReport
from leann_sources.manifest import SourceManifest
from leann_sources.readers.api import APISourceReader


class GitHubSourceReader(APISourceReader):
    def __init__(self, manifest: SourceManifest):
        super().__init__(manifest)

    def validate(self) -> ValidationReport:
        base_report = super().validate()
        if not base_report.ok:
            return base_report
        if not self._repositories() and self.manifest.data.get("pages") is None:
            return ValidationReport(
                False,
                "missing config",
                errors=[
                    "set GITHUB_REPOSITORIES to one or more owner/repo values, or declare data.repositories"
                ],
            )
        return ValidationReport(True, "ok")

    def iter_records(self, since: datetime | None = None) -> Iterator[dict[str, Any]]:
        fixture_pages = self.manifest.data.get("pages")
        if fixture_pages is not None:
            for page in fixture_pages:
                for record in page:
                    yield self._normalize_record(record)
            return

        token = os.environ.get("GITHUB_TOKEN", "")
        self.session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "Authorization": f"Bearer {token}",
            }
        )
        since_param = since.isoformat() if since else None
        for repository in self._repositories():
            for record in self._fetch_repository_records(repository, since=since_param):
                yield self._normalize_record(record, repository=repository)

    def _repositories(self) -> list[str]:
        repositories = self.manifest.data.get("repositories")
        if repositories:
            return [
                str(repository).strip() for repository in repositories if str(repository).strip()
            ]
        env_var = self.manifest.data.get("repositories_env", "GITHUB_REPOSITORIES")
        raw_value = os.environ.get(env_var, "")
        return [part.strip() for part in raw_value.split(",") if part.strip()]

    def _fetch_repository_records(
        self, repository: str, *, since: str | None
    ) -> Iterator[dict[str, Any]]:
        for path, kind in (
            ("issues", "issue"),
            ("issues/comments", "issue_comment"),
            ("commits", "commit"),
        ):
            url = f"{self.manifest.data.get('base_url', 'https://api.github.com')}/repos/{repository}/{path}"
            params: dict[str, str] = {"state": "all", "per_page": "100"}
            if since:
                params["since"] = since
            while url:
                response = self.session.get(
                    url, params=params, timeout=self.manifest.data.get("timeout", 30)
                )
                response.raise_for_status()
                remaining = response.headers.get("X-RateLimit-Remaining")
                self.rate_limit_remaining = (
                    int(remaining) if remaining and remaining.isdigit() else None
                )
                for record in response.json():
                    record["_kind"] = (
                        "pull_request" if kind == "issue" and record.get("pull_request") else kind
                    )
                    record["_repository"] = repository
                    yield record
                url = response.links.get("next", {}).get("url")
                params = {}

    def _normalize_record(
        self, record: dict[str, Any], repository: str | None = None
    ) -> dict[str, Any]:
        kind = record.get("_kind") or record.get("kind") or "issue"
        repository = repository or record.get("_repository") or record.get("repository") or ""
        if kind == "commit":
            commit = record.get("commit") or {}
            author = commit.get("author") or {}
            committer = commit.get("committer") or {}
            message = commit.get("message") or ""
            sha = record.get("sha") or record.get("node_id") or ""
            files = [
                file.get("filename", "") for file in record.get("files", []) if file.get("filename")
            ]
            return {
                "kind": "commit",
                "title": message.splitlines()[0] if message else sha,
                "text": message,
                "created_at": author.get("date") or committer.get("date"),
                "updated_at": committer.get("date") or author.get("date"),
                "event_time": author.get("date") or committer.get("date"),
                "author": (author.get("name") or author.get("email") or "unknown"),
                "source_type": "github",
                "source_id": f"github:{repository}:commit:{sha}",
                "source_url": record.get("html_url", ""),
                "project_id": repository,
                "parent_ref": f"repo:{repository}",
                "source_document_id": f"github:{repository}:commits",
                "mentioned_files": files,
            }

        user = record.get("user") or {}
        number = record.get("number")
        title = record.get("title") or f"{kind} {record.get('id', '')}".strip()
        body = record.get("body") or record.get("text") or ""
        source_id = record.get("node_id") or record.get("id") or record.get("url") or title
        issue_url = record.get("issue_url") or record.get("html_url", "")
        parent_ref = f"repo:{repository}"
        source_document_id = f"github:{repository}:{kind}s"
        if number:
            parent_ref = f"github:{repository}#{number}"
            source_document_id = f"github:{repository}:issue:{number}"
        return {
            "kind": kind,
            "title": title,
            "text": body,
            "created_at": record.get("created_at"),
            "updated_at": record.get("updated_at") or record.get("created_at"),
            "event_time": record.get("created_at"),
            "author": user.get("login") or record.get("author") or "unknown",
            "source_type": "github",
            "source_id": f"github:{repository}:{kind}:{source_id}",
            "source_url": record.get("html_url") or issue_url,
            "project_id": repository,
            "parent_ref": parent_ref,
            "source_document_id": source_document_id,
            "mentioned_files": record.get("mentioned_files", []),
        }
