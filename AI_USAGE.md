# AI Usage Record

This project used AI assistants as development and research-support tools. AI output was treated as a draft or implementation aid and was reviewed by the student/team before being retained. Experimental claims and numerical results reported in this repository come from completed local executions and saved artifacts, not from numbers supplied by an AI assistant.

## Tools used

| Tool | Use in the project | Verification / modification |
|---|---|---|
| Perplexity / ChatGPT-like assistant | Paper interpretation, research planning, method review, dataset/provenance reasoning, comparison framing, and prompt drafting | Paper statements were checked against the available paper extraction and project evidence. The student/team decided which claims were supportable and edited the final documentation. |
| Claude / coding assistant | Repository scaffolding, implementation support, test generation, debugging support, documentation drafts, and repository packaging | Code and tests were inspected; commands were run locally; exit codes and saved outputs were checked; artifact hashes and real-data counts were independently verified before results were retained. |

## High-level prompt log

| Date | Tool | High-level request | Output/use | Verification / modification |
|---|---|---|---|---|
| 2026-09-24 | Claude | Scaffold the replication repository, configuration, assumptions, paper extraction, protocol, and README | Initial project structure and documentation drafts | Student/team reviewed paper-specific claims, assumptions, and repository structure and retained only documented choices. |
| 2026-09-24 to 2026-09-25 | ChatGPT-like assistant | Analyze paper/provenance, reconcile the public ADFA-LD mirror with the paper, and plan reproducible reconstruction stages | Dataset forensic reasoning, reconstruction protocol, XAI/perturbation review, and implementation prompts | Counts, hashes, tests, and experiment outputs were checked against local files; unsupported claims were removed or labelled as limitations. |
| 2026-09-25 | Claude / coding assistant | Implement and harden frozen reconstruction, XAI, and perturbation stages and their tests | Source/test/documentation changes and experiment orchestration | Student/team inspected implementation, ran the test suite and experiment commands, checked exit codes, reviewed outputs, and corrected documentation/bookkeeping issues before freezing the results. |
| 2026-09-25 | ChatGPT-like assistant | Finalize Phase 1 documentation without changing frozen experimental artifacts | Result comparison, one-page summary, LIME verification plan, README, and AI-use documentation | Documentation was reconciled against the frozen metrics, hashes, paper extraction, and current repository contents. |

## Human/team responsibility

The student/team remained responsible for:

- prompting and task decomposition;
- inspecting implementation output;
- reviewing source code and tests;
- running commands and validating exit codes;
- checking artifact hashes;
- verifying real-data counts and dataset inventory;
- deciding which implementation/output changes to retain;
- reviewing limitations and provenance claims;
- editing and approving the final documentation.

## Experimental evidence rule

No reported experimental number is accepted merely because an AI assistant produced it. Reported metrics, confusion matrices, explanation statistics, perturbation statistics, and hashes must come from completed local executions or verified saved artifacts in the repository.

## LIME disclosure

The completed Phase 3B/3C work used a vendored LIME-compatible local-surrogate implementation because the official `lime==0.2.0.1` package could not be installed in the offline environment. The project does not claim that the official external package was executed.
