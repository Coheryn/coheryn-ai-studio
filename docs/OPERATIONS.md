# CPU operation and recovery

Use the experimental CPU build only within the explicit [compatibility scope](../fixtures/compatibility.json). Production approval is tracked in [CPU_RELEASE.md](CPU_RELEASE.md). GPU execution is not part of the CPU candidate. Model artifacts are acquired separately with their expected checksums and applicable model licenses.

## Start and observe

Run `coheryn doctor` to inspect compiled capabilities. Keep the data directory on a local filesystem under a dedicated account. On Unix it must be private (0700); credential files are private (0600). One daemon owns a data directory. Keep model/checkpoint files immutable while in use and separate them from untrusted writers.

```sh
coheryn --data-dir /absolute/private/coheryn serve --max-concurrency 1
coheryn --data-dir /absolute/private/coheryn status --json
```

The daemon binds loopback by default and creates distinct admin/inference credentials. Never place tokens in shell command arguments, logs or issue reports. Remote exposure requires deliberate binding, TLS termination, role separation and reviewed ingress/rate limits. Consult [API.md](API.md) for exact flags, credentials, routes and limits; the packaged CLI's `--help` is the authority for the installed binary.

Inspect and deeply validate a model before loading it. Use `plan` to estimate the requested context, profile, concurrency and image/embedding stages. Explicit load is required for daemon inference. `--standalone` creates a separate local instance and adds its own memory demand; the client never silently falls back to it.

`--max-memory` controls engine reservations/admission. It is not a process RSS limit. Compare status reservations and buffer capacities with OS observations, leaving headroom for tokenizers, HTTP, SQLite and the OS. Apply an OS service/container memory limit when deployment requires a hard process boundary. Requests that cannot fit must fail explicitly; never silently reduce context or swap the model/backend.

Monitor queue depth, active requests, errors, cancellation cleanup, memory pressure and request-stage timings. Keep prompts, images, tool arguments, credentials and complete session exports out of operational logs. Telemetry is disabled. Cache conditions and load/tokenization/prefill/decode/delivery timings must remain explicit in performance reports.

## Stop, backup and restart

Cancel/drain active requests and stop the daemon normally before maintenance. Confirm that the process exited before copying, restoring or modifying its data directory. Persistent history is opt-in; ephemeral sessions do not survive restart. Do not promise completion of an interrupted generation. Completed durable receipts retain idempotency; a recovered pending writer is marked cancelled rather than silently regenerated.

Back up the entire stopped data directory, including registry, model/checkpoint assets, history database/assets and credential vault, as one consistent protected snapshot. Session exports are a portable conversation interface, not a replacement for complete registry/assets/credentials backups. Encrypt backups and restrict access: they may contain original user images and text. Do not copy a live database file independently of related assets or journal state.

Test restoration into a separate private directory while the original daemon is stopped. Verify model hashes, history ownership, aliases, completed receipt replay and cancelled pending turns before resuming traffic. Never run two daemons against the same directory. The process-kill tests exercise SQLite/application restart boundaries; power loss, network filesystems and disk failure need separate deployment qualification.

## Upgrade and rollback

Verify the archive checksum from a trusted channel and review signing/provenance before execution. Preserve the previous executable and a stopped, consistent data snapshot. Test the new binary against a copy first, including schema compatibility, model validation, representative text/image/embedding requests, streaming and restart/replay.

If rollback is needed, stop the new daemon and restore both the previous binary and its matching data snapshot. Do not point an older binary at a newly migrated store unless that version explicitly documents backward compatibility. Preserve failed-request diagnostics without exposing secrets or original user content. Recovery must use documented APIs/backups; manually editing live registry or history records is unsupported.
