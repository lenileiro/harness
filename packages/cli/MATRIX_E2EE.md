# Matrix end-to-end encryption

Matrix encryption is opt-in. Harness uses
[`matrix-nio` 0.26.0](https://pypi.org/project/matrix-nio/0.26.0/), whose pinned
`e2e` dependency uses the Rust `vodozemac` implementation. Install the optional
CLI extra with `uv sync --package cli --extra matrix`. Wheels supply the native
crypto library on supported platforms; a platform without a compatible
vodozemac wheel needs its Rust/PyO3 build toolchain. This SDK version does not
require a separate libolm installation. Missing crypto dependencies fail closed.

```toml
[channels.matrix]
homeserver = "https://matrix.example.org"
token_env = "MATRIX_BOT_TOKEN"
profile = "matrix-bot"
allowed_users = ["@owner:matrix.example.org"]
allowed_channels = ["!private-room:matrix.example.org"]
matrix_e2ee = true
matrix_device_id = "HARNESS_DEVICE"
matrix_store_path = "/absolute/private/path/matrix-keys"
matrix_pickle_key_env = "MATRIX_PICKLE_KEY"
matrix_unverified_policy = "reject"

[channels.matrix.matrix_trusted_devices."@owner:matrix.example.org"]
OWNER_DEVICE = "EXACT_ED25519_FINGERPRINT_FROM_VERIFIED_DEVICE"
```

Use a dedicated Matrix device/access token, join the intended room, and enable
encryption there with your normal Matrix client. The token's actual device ID
must match `matrix_device_id`. Supply a stable, private pickle passphrase of at
least 32 characters through its environment reference; no credential values are
written to configuration. Run with the matching profile:

```sh
harness --profile matrix-bot channel run matrix
```

Device fingerprints are an explicit operator decision. Obtain each intended
device's Ed25519 fingerprint through an independent verified channel, then put
that exact value in `matrix_trusted_devices`. Spaces in displayed fingerprints
are accepted. New, changed, unknown, or unverified devices block outbound
delivery. Every current recipient device, including the bot account's other
devices, must be pinned. Harness never enables blanket trust or silently ignores
unverified devices. This is manual device verification; the SDK does not provide
cross-signing or server-side secure key backup. Keep those limitations in mind
when selecting a Matrix client for key recovery.

The private store is separated by homeserver, account, device, and profile.
Its directory must have mode 0700 on POSIX; stored files are private, symlinks
and special files are refused, and one process holds the device-store lock.
SDK account and inbound session keys persist across restarts, encrypted using
the referenced pickle passphrase. Back up the store and passphrase together.
If a new empty store disagrees with the server's existing device keys, startup
refuses to overwrite that identity: restore the original store or create a new
device and token. Do not reuse a device identity for several independent stores.

In encryption mode, outbound text and thread relations are encrypted before a
room-send request can reach HTTP. The HTTP adapter itself rejects plaintext
room-send endpoints. Media bytes are encrypted before upload; the media
repository receives ciphertext and a generic filename, while the encryption
metadata and original filename travel inside the encrypted room event. Downloads
verify the encrypted attachment hash before exposing decoded bytes. An
unencrypted or unsynced room never triggers plaintext fallback.

Inbound plaintext events are ignored in encryption mode. Only messages decrypted
by the SDK and verified against the configured sender-device pins enter the
Harness channel inbox, where existing user/room/group/mention policies still
apply. Missing or unverified keys leave ciphertext in a bounded durable queue
(256 events / 16 MiB), so keys arriving later or a process restart do not erase
the message. Existing first-sync history suppression and event-ID deduplication
remain in effect. SDK requests do not automatically retry uncertain sends;
inspect the destination before using the channel's explicit delivery retry.

Offline tests run two real SDK crypto devices against a local HTTPX mock
homeserver. They exercise signed one-time-key exchange, Olm session sharing,
Megolm encryption/decryption, inbound-key restoration, delayed keys, device
trust failures, media integrity, device locking, and interrupted inbox commits.
No test contacts a live account or room.

References: [SDK API](https://matrix-nio.readthedocs.io/en/latest/nio.html),
[Matrix encryption guide](https://matrix.org/docs/matrix-concepts/end-to-end-encryption/).
