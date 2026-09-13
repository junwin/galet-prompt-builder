# Prompt compilation design

Status: initial implementation

## Purpose

Given explicit token budgets, configured hard maxima, caller-supplied prompt
instructions, and the CoALA interfaces from `galet-memory`, compile a bounded
provider-neutral prompt and explain the result through metrics.

The package owns orchestration, selection, budgeting, relevance assessment,
rendering, and metrics. It does not own agents, tools, request handling,
storage construction, authentication, or provider routing.

## Invariants

- System instructions and current input are mandatory.
- Mandatory content is never silently truncated or removed.
- Compilation fails when mandatory content cannot fit the usable total limit.
- Requested budgets cannot exceed application-controlled maxima.
- Each memory section has an independent hard token ceiling.
- The rendered prompt stays below the total budget minus its safety margin.
- Compilation does not create digests or mutate any memory source.
- Metrics contain counts and sizes, not copied memory content.
- No LLM call is implicit. The default relevance policy is deterministic.
- Optional LLM relevance uses only Galet's `LLMApi`.

## Inputs

`PromptRequest` carries caller-owned instructions, the current input, account
and memory selectors, and inclusion flags. It contains no agent type.

`PromptBudgets` describes the requested total, safety margin, and independent
budgets for procedural memory, episodic events, episodic digests, and semantic
documents.

`PromptLimits` is controlled by the host and caps every requested budget plus
retrieval counts, item size, and message-overhead accounting.

## Retrieval

The compiler calls only the `ProceduralMemory`, `EpisodicMemory`, and
`SemanticMemory` interfaces. It requests procedural memory without creating a
missing context. Episodic recall receives an empty `agent_name`; the compiler
does not perform agent resolution. Semantic retrieval uses the current input
unless the caller supplies a distinct semantic query.

Memory failures are explicit `MemoryRetrievalError` failures. A future policy
may support best-effort omission, but that must be requested rather than
silently producing an incomplete prompt.

## Digests

The current session boundary digest arrives among active episodic events and
is treated as required within the episodic-event budget. Relevant historical
digests use the separate digest budget. Matching digest text is deduplicated
so the same digest is not inserted through both paths.

Prompt compilation never generates or stores an overflow digest. Digest
production and curation belong to `galet-memory` and its host adapters.

## Selection

The default global memory priority is:

1. Procedural memory.
2. Active episodic events.
3. Semantic documents.
4. Historical episodic digests.

Within a section, required items precede relevance and recency. Episodic
events are rendered back into chronological order after selection. An item
that cannot fit the remaining section allowance may be truncated; lower-ranked
items are then omitted. Unused budgets are not transferred between sections
in this first version.

## Output and metrics

`CompiledPrompt` contains structured `PromptMessage` values, provider-shaped
role/content dictionaries, a plain-text rendering, and `PromptMetrics`.

Metrics report requested budget, tokens used, items retrieved, selected,
dropped and truncated for each section, together with total usage and the
safety margin. Provider-specific token counters can replace the deterministic
approximation through the `TokenCounter` port.

## Future questions

- Whether unused section budgets should be redistributed.
- Whether applications need a best-effort memory failure policy.
- How multimodal inputs and tool schemas participate in total budgeting.
- Whether relevance thresholds should be caller-configurable.
- Which provider-specific token counters should be supplied as adapters.
