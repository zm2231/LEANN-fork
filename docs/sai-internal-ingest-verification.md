# SAI internal-ingest verification

The canonical post-P3 operator contract for SAI internal-ingest retrieval verification
lives in the SAI repository at:

```text
docs/internal-ingest-verification-operator-guide.md
```

Use that document rather than reproducing the release schema or commands in LEANN docs.
The standing SAI harness is `tests/internal_ingest_verification.py`, accepted at SHA-256
`32e44f9fb384a7ad0651fa15181cf32364369330a5db7be0a6797759b7d67740`.

LEANN supplies the hash-pinned executable and existing index artifacts referenced by an
accepted release. The SAI gate does not build, rebuild, remove, or mutate LEANN indexes.
Its `fable_p3` fields are shape-checked by code; Fable P3 verifies that the cited receipt
is real, accepted, and authorizes the exact batch before execution.
