# ADK repair checklist — approved for release

- [x] Replace handwritten HTTP with the published asynchronous GoodMem SDK.
- [x] Isolate scope caches by component and connection; honor explicit configuration.
- [x] Preserve all matching chunks and join actual memory definitions.
- [x] Preserve statuses, partial results, and errors, including unknown codes and corrupt streams.
- [x] Verify callbacks and tools leave the event loop responsive.
- [x] Report accepted IDs and attachment failures honestly.
- [x] Make live cleanup reliable and replace permissive recall assertions.
- [x] Update examples, migration guidance, packaging, and CI.
- [x] Add the missing ADK entry to the recurring integration harness.
- [x] Run installed-distribution regressions and live GoodMem/Cohere journeys.
- [x] Replace the internal service's global ingestion lock with session-level locking.
- [x] Share attachment validation/uploads and preserve accepted writes across retries.
- [x] Correct and execute the SDK migration example.
- [x] Freeze the 0.1.1 audit independently of the current test suite and replay it.
- [x] Add live concurrent-ingestion and partial-upload recovery tests to the recurring suite.
- [x] Review the complete diff and receive Amin's approval to commit and publish.

Baseline evidence remains in REPORT.md. Candidate validation and review decisions
are summarized in [REVIEW.md](REVIEW.md).
