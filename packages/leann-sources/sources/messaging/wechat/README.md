# WeChat Export Source

Indexes JSON exports produced by the local WeChat export tooling.

```bash
leann index --source wechat
leann index-wechat --export-dir /path/to/wechat_export
```

The reader preserves the existing grouped-message behavior and emits event
time from message creation time, with modified time from the grouped end time.
