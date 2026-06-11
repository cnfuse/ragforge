# ADR 0001 — Record architecture decisions

- **Status:** Accepted
- **Date:** 2026-06-12

## Context

The project makes several non-obvious design choices (offline-by-default
dependencies, an agentic loop rather than context stuffing, protocols over
inheritance). Future readers — including hiring reviewers — benefit from
understanding *why*, not just *what*.

## Decision

Use lightweight Architecture Decision Records (ADRs), one Markdown file per
decision in `docs/adr/`, numbered sequentially. Each ADR states context,
decision, and consequences. ADRs are immutable once accepted; a reversal is a new
ADR that supersedes the old one.

## Consequences

- A durable, reviewable history of *why* the system looks the way it does.
- Minimal overhead — a short file per significant decision.
