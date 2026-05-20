# Chrome History Source

Indexes Chrome browser history from the profile `History` SQLite database.

Default profile:

```bash
~/Library/Application Support/Google/Chrome/Default
```

The legacy `index-browser brave` alias is preserved by overriding the source
path to the Brave profile before dispatching through this source.

```bash
leann index --source chrome
leann index-browser
```
