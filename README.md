# Shopping Copilot AI Conversational Search and Recommendations

Project files are organized as independent modules:

```text
intent-recognition/          current-message intent and query routing
retrieval-and-reranking/     complete-requirements Top50 and Top10 ranking
synthetic-data-3021/         local proxy sessions for stress testing
```

## Integration boundary

`intent-recognition/` parses one message at a time. A separate state or dialogue
component must accumulate the disclosed category, hard constraints, and soft
preferences. Only after those requirements are considered complete should they
be passed to `retrieval-and-reranking/`.

The current retrieval results therefore measure a **complete-information
condition**. They do not prove that two clarification questions will recover all
requirements in the organizer's private sessions.

## Collaboration

Please create a feature branch, run the tests for the module you changed, and
open a pull request instead of pushing directly to `main`. See
[`CONTRIBUTING.md`](CONTRIBUTING.md) for the project workflow and large-file
rules.
