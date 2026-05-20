# WhatsApp Source

The WhatsApp source reads a local `ChatStorage.sqlite` database, usually copied from an iOS backup.

Typical setup:

1. Create an encrypted iOS backup from Finder.
2. Extract the WhatsApp `ChatStorage.sqlite` file with a backup browser.
3. Run `leann sources validate whatsapp` after setting `data.default_path` or invoking `leann index --source whatsapp ... --path`.

The reader emits message-level chunks with WhatsApp message time as `event_time`, synthesized `created_at`, edit time as `modified_at` when available, participant handles, URLs, and thread document IDs.
