# GitHub Source

The GitHub source reads issues, pull requests, issue comments, and commits through the GitHub REST API.

Configuration:

- `GITHUB_TOKEN`: personal access token or fine-grained token with read access to the repositories.
- `GITHUB_REPOSITORIES`: comma-separated `owner/repo` list, unless `data.repositories` is set in the manifest or local config.

Run `leann sources validate github` before indexing. Live indexing is intentionally manual because repository selection and token scope are user-specific.
