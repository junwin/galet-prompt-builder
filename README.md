# galet-prompt-builder

Provider-neutral orchestration of CoALA memory into bounded prompts.

The package compiles caller-supplied system instructions and current input
with procedural, episodic, and semantic memory from `galet-memory`. It applies
independent section budgets, a total hard limit, relevance assessment, digest
deduplication, deterministic selection, and detailed metrics.

It deliberately has no concept of Lucy agents, handlers, storage paths,
attachments, or application configuration. Lucy can later adapt those values
into `PromptRequest`, `PromptBudgets`, and `PromptLimits`.

## Basic use

```python
from galet_prompt_builder import (
    PromptBudgets,
    PromptCompiler,
    PromptLimits,
    PromptRequest,
)

compiler = PromptCompiler(
    procedural_memory=procedural_memory,
    episodic_memory=episodic_memory,
    semantic_memory=semantic_memory,
)

compiled = compiler.compile(
    PromptRequest(
        account_name="demo",
        conversation_id="session-id",
        context_name="project",
        system_instructions=["You are a careful assistant."],
        current_input="What did we decide about storage?",
        semantic_namespaces=("documents",),
        semantic_score_threshold=0.30,
        episodic_event_kinds=(
            "user_message",
            "assistant_message",
            "session_digest",
        ),
        include_structured_episodic_events=False,
        episodic_digest_score_threshold=0.40,
    ),
    PromptBudgets(
        total_tokens=8000,
        procedural_tokens=1500,
        episodic_event_tokens=2000,
        episodic_digest_tokens=1000,
        semantic_tokens=2000,
        safety_margin_tokens=500,
    ),
    PromptLimits(
        maximum_total_tokens=16000,
        maximum_procedural_tokens=3000,
        maximum_episodic_event_tokens=4000,
        maximum_episodic_digest_tokens=2000,
        maximum_semantic_tokens=4000,
        maximum_events=6,
        maximum_digests=2,
        maximum_semantic_documents=3,
        maximum_episodic_event_chars=4000,
        maximum_episodic_digest_chars=1800,
        maximum_semantic_item_chars=1800,
    ),
)

provider_messages = compiled.provider_messages
metrics = compiled.metrics
```

The default relevance assessor is deterministic and makes no network calls.
`GaletRelevanceAssessor` is available when an application explicitly wants an
LLM-based assessment; all such calls go through Galet's `LLMApi`.

Prompt policy is explicit and agent-independent. Applications choose the
overall and per-section token budgets, retrieval counts, event kinds, relevance
thresholds, and per-item character caps. An agent may supply system
instructions, but no agent object or application policy is required.


## Prompt comparison CLI

The installed `galet-prompt-run` command compiles a prompt from an existing
Lucy chat and a new request. It resolves the chat's friendly name to its stored
session ID and reads episodic history from Lucy's SQLite database.

By default it reads:

```text
/home/junwin/lucy_storage/data/chat2.sqlite
```

Example:

```bash
galet-prompt-run \
  "What should we do next?" \
  --chat-name "Prompt Builder Work"
```

Use a different storage root or an explicit database when needed:

```bash
galet-prompt-run \
  "Compare this prompt" \
  --chat-name "Prompt Builder Work" \
  --storage-root /srv/lucy_storage

galet-prompt-run \
  "Compare this prompt" \
  --chat-name "Prompt Builder Work" \
  --db /tmp/chat2.sqlite \
  --format json
```

The command is read-only. It fails if the database or friendly chat name does
not exist, and it reports duplicate friendly names instead of guessing. The
runner includes episodic events and digests by default.

Semantic recall is opt-in. Supply one or more namespaces to query Lucy's
`embeddings-v2.sqlite` store:

```bash
galet-prompt-run \
  "What did I write about attention?" \
  --chat-name "Prompt Builder Work" \
  --namespaces vol_6 vol_7 documents
```

The comparison runner uses deliberately compact defaults:

- 8,000 total tokens with a 500-token safety margin
- 1,000 recent-event tokens across at most 6 events
- 500 digest tokens across at most 2 digests
- 1,000 semantic tokens across at most 3 documents
- semantic score threshold 0.30
- digest score threshold 0.40
- conversational event kinds only; structured tool payloads are excluded

Every policy value can be stated explicitly:

```bash
galet-prompt-run \
  "What did I write about attention?" \
  --chat-name "Prompt Builder Work" \
  --namespaces vol_6 vol_7 documents \
  --total-tokens 6000 \
  --safety-margin-tokens 500 \
  --episodic-event-tokens 800 \
  --max-events 6 \
  --episodic-digest-tokens 400 \
  --max-digests 2 \
  --digest-score-threshold 0.45 \
  --semantic-tokens 900 \
  --semantic-top-k 3 \
  --semantic-score-threshold 0.35 \
  --semantic-item-max-chars 1800
```

With no `--namespaces`, the namespace list is empty and the runner does not
open the embedding database or make an embedding API call. Semantic mode
defaults to
`/home/junwin/lucy_storage/data/embeddings-v2.sqlite`; use
`--embedding-db` to override it and install the optional dependency with
`pip install ".[semantic]"`. Galet supplies the query embedding, using its
normal credential configuration or the directory passed to
`--credential-path`.

Procedural memory remains disabled until its context repository is configured
explicitly.

## Dependency rule

> Applications may depend on `galet-prompt-builder`; the package must never
> depend on an application.

The package depends on the neutral interfaces from `galet-memory`. It never
opens memory databases itself and prompt compilation never mutates memory.
