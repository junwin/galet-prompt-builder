# galet-prompt-builder

Provider-neutral orchestration of CoALA memory into bounded prompts.

The caller supplies instructions, current input, retrieval options, and
budgets. Galet-prompt-builder obtains memory through galet-memory, chooses
relevant candidates within the limits, and assembles provider messages with
selection metrics. It consumes digests; it does not generate or store them.
The host application chooses when to compile a prompt and which policy to use.

See the [galet architecture notes](https://github.com/junwin/galet/blob/main/docs/architecture.md)
for the full layering.

## Install

Requires Python 3.11 or later. The commands below use a Linux/macOS shell.

### Install the published package

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install galet-prompt-builder
```

Both `galet` and `galet-memory` are installed automatically as required
dependencies. Add `[semantic]` when using the SQLite vector store:
`python -m pip install "galet-prompt-builder[semantic]"`.

### Install from source and run the tests

```bash
git clone --branch develop https://github.com/junwin/galet-prompt-builder.git
cd galet-prompt-builder
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[test,semantic]"
python -m pytest -q
```

For an existing checkout, start from `cd galet-prompt-builder` and reuse
your `.venv` if present. The `test` extra installs pytest; `semantic` installs
sqlite-vec for vector recall. You can omit `semantic` if you do not use it.

Editable installation (`-e`) means Python imports this checkout, so source
changes take effect without reinstalling. Dependencies come from your
configured package index. To develop against sibling source checkouts too,
install them into this same environment:

```bash
python -m pip install -e ../galet -e ../galet-memory
```

Those paths assume all three repositories share a parent directory.
No `PYTHONPATH` setting is needed after the editable installation.

### Update an existing source installation

Update your checkout with Git, activate its virtual environment, then run:

```bash
python -m pip install --upgrade --upgrade-strategy eager -e ".[test,semantic]"
python -m pytest -q
```

This upgrades dependencies within the constraints in `pyproject.toml`.
If you are developing against editable sibling checkouts, update those
repositories separately too.

## Shared configuration (recommended)

Prompt-builder settings contain prompt policy and a reference to the shared
**galet-memory config**, not another set of store paths or credential values.
`galet_prompt_builder_sample.json` is a complete example. Its relative
`memory_config_path` points to the sibling galet-tools config already tested
through MCP; adjust the reference to your actual file.

```json
{
  "config_version": 1,
  "memory_config_path": "../galet-tools/galet-memory-mcp.json",
  "policy": {
    "total_tokens": 6000,
    "semantic_tokens": 900,
    "maximum_semantic_documents": 3
  }
}
```

The public loader selects a configuration object, an explicit `path`, or
`GALET_PROMPT_BUILDER_CONFIG_PATH`, in that order. Sources are selected rather
than merged. File references resolve relative to the containing prompt config;
a configuration object with a relative reference requires `base_dir`.
Without prompt settings, policy defaults apply and galet-memory selects
`GALET_MEMORY_CONFIG_PATH`. An invalid selected source fails without falling back.
Galet-memory still owns its own config validation and backend assembly. Galet
still owns credential lookup (`GALET_CREDENTIAL_PATH` or the selected profile).
The configured embedding model is used for retrieval, rather than a hardcoded
query model. Requires galet-memory 0.1.28 or newer.

When semantic retrieval is enabled, an embedding-provider failure raises
`MemoryRetrievalError`, including when galet-memory offers keyword fallback.
This prevents missing credentials or an unavailable provider from appearing as
a successful search with no relevant memory. Standalone galet-memory callers
can still use its explicit keyword fallback. Successful searches with zero
matches remain valid. Disabling semantic retrieval does not require embeddings
for that channel; digest retrieval can independently need embeddings.

The `galet_prompt_builder.compiler` logger reports semantic backend and document
counts, followed by retrieved/selected/dropped counts and policy warnings after
compilation. Enable INFO logging in the host application. Failure records use
ERROR level. The package does not install handlers or change global log levels.

### Compile against the same existing stores as MCP

```bash
export GALET_CREDENTIAL_PATH=/home/junwin/credential
python -m galet_prompt_builder.cli "What do my notes say about Jon Fosse?" \
  --config galet_prompt_builder_sample.json \
  --account junwin --session-id YOUR_EXISTING_SESSION_ID \
  --namespaces books --format json
```

Or skip the policy file and pass the shared memory config directly:

```bash
python -m galet_prompt_builder.cli "What happened at Tyntesfield?" \
  --memory-config /home/junwin/src/repos/galet-tools/galet-memory-mcp.json \
  --account junwin --session-id YOUR_EXISTING_SESSION_ID --namespaces vol_7
```

`--chat-name` can replace `--session-id`. Configured mode uses the declared
semantic metadata version, episodic database and procedural layout; it does not
infer any of them from filenames. The comparison CLI requires existing stores
and will not create missing databases, even if a shared test config has
`create:true`. Legacy storage/layout/provider flags cannot be combined with
shared configuration. Policy flags such as `--total-tokens` override only the
specified policy values. Use `--no-semantic` or `--no-digests` to disable those
retrieval channels. No LLM reply is generated, but semantic/digest retrieval can
make paid embedding requests.

When configured namespaces are omitted, retrieval uses the named procedural
context's `search_namespaces`, then all namespaces belonging to the account if
none are declared. Explicit namespaces remain authoritative.

### Python API and runtime ownership

```python
from galet_prompt_builder import build_prompt_builder, PromptRequest

with build_prompt_builder(path="galet_prompt_builder_sample.json",
                          allow_create=False) as builder:
    compiled = builder.build(PromptRequest(
        account_name="junwin",
        conversation_id="YOUR_EXISTING_SESSION_ID",
        context_name="reading",
        current_input="What do my notes say about Jon Fosse?",
        semantic_namespaces=("books",),
    ))
    print(compiled.provider_messages)
    print(compiled.metrics)
```

The factory owns and closes memory it constructs. An application that already
owns a runtime can borrow it instead, avoiding duplicate stores:

```python
with build_prompt_builder({"config_version": 1},
                          memory_runtime=runtime) as builder:
    compiled = builder.build(request)
# runtime remains open and owned by the application.
```

Borrowing cannot be combined with another memory-config reference or provider
factory. `embedding_provider_factory` remains available when constructing memory
for tests or explicit host integrations. The factory normally honors the memory
config's creation flags; `allow_create=False` disables creation for existing-store
inspection. Neither prompt construction nor compilation ingests documents,
creates sessions, publishes digests, or changes events.

For the Python API, an empty `semantic_namespaces=()` opts into context/all-account
selection. `PromptRequest` retains its older default `("external",)` for existing
callers. `PromptCompiler` still accepts explicitly injected memory interfaces.

Configured digest retrieval resolves V3 root-relative locators and checks account
ownership and invalidation before reading a file. Legacy metadata paths are used
only in explicit V2 mode and remain confined to configured roots. Missing or
unauthorized sources do not expose stored text snapshots. A borrowed runtime's
explicit digest-search provider is preserved. Working state can be part of the
shared runtime but is not inserted into prompts automatically.

### Application boundary

Prompt-builder owns retrieval, policy, selection and rendering. This configured
entry point means a host need not construct memory backends or adapt query models
just to compile a prompt. Session creation, event append, archive/reset and digest
publication remain application/memory operations. They are not moved into the
prompt compiler. Applications may retain direct galet-memory use for those tasks
and lend the same runtime to prompt-builder. No Lucy-specific imports or global
runtime are introduced here.

## Run locally

Keep the virtual environment active for the commands below. In a new shell,
run `source .venv/bin/activate` from the checkout first.

### Compile a prompt from Python

This complete example needs no database, credentials, or network calls. It
compiles the supplied instructions and input without attaching memory stores:

```bash
python - <<'PY'
from galet_prompt_builder import PromptCompiler, PromptPolicy, PromptRequest

compiled = PromptCompiler().build(
    PromptRequest(
        account_name="demo",
        system_instructions=("You are a careful assistant.",),
        current_input="Explain what a prompt compiler does.",
        include_procedural=False,
        include_episodic=False,
        include_semantic=False,
        include_digests=False,
    ),
    PromptPolicy(),
)

print(compiled.text)
print(compiled.provider_messages)
print(compiled.metrics)
PY
```

This prints the assembled prompt and metrics; it does not send the prompt to
an LLM. For memory-backed use, construct and pass the stores shown in
[Basic use](#basic-use).

### Legacy comparison CLI without shared configuration

The editable installation also installs `galet-prompt-run`. Check its options:

```bash
python -m galet_prompt_builder.cli --help
```

To compile against your existing chat database, replace the account and chat
name below with your stored values:

```bash
python -m galet_prompt_builder.cli "What should we do next?" \
  --account your-account \
  --storage-root "$HOME/lucy_storage" \
  --chat-name "Your existing chat name"
```

You can replace `python -m galet_prompt_builder.cli` with `galet-prompt-run`;
both invoke the same runner. It expects an existing compatible
`<storage-root>/data/chat2.sqlite` database and an existing session; installation
does not create sample chat data. Use `--db /path/to/chat2.sqlite` to override
the chat database and `--storage-namespace` if it is not `data`.

Pass your own account and storage root explicitly: the CLI currently defaults
to account `junwin` and storage root `/home/junwin/lucy_storage`.
The CLI prints a compiled prompt and metrics; it does not generate an LLM reply.
See [Prompt comparison CLI](#prompt-comparison-cli) for memory and policy options.

When an embeddings database exists at the selected path, the runner also
initializes vector recall for digests, even without `--namespaces`. That path
needs the `semantic` extra and Galet embedding credentials; use
`--credential-path /path/to/credentials` to supply the credential directory.
Adding `--namespaces` also enables semantic document recall.

The package compiles caller-supplied system instructions and current input
with procedural, episodic, and semantic memory from `galet-memory`. It applies
independent section budgets, a total hard limit, relevance assessment, digest
deduplication, deterministic selection, and detailed metrics.

The core compiler has no concept of host agents, handlers, attachment storage,
or application-specific configuration. The configured facade above handles
only prompt policy and the shared memory reference. The host adapts other values into
`PromptRequest`, `PromptBudgets`, and `PromptLimits`.

## Basic use

The examples below assume you have already constructed `procedural_memory`,
`episodic_memory`, and `semantic_memory` stores for your application. Use the
self-contained example in [Run locally](#run-locally) to verify installation first.

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
    digest_memory=episodic_memory,  # only when its digest_search provider is configured
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

Applications can pass the same knobs used by `galet-prompt-run` through
`PromptPolicy`, without constructing budgets and limits themselves:

```python
from galet_prompt_builder import PromptCompiler, PromptPolicy, PromptRequest

compiled = PromptCompiler(
    episodic_memory=episodic_memory,
    digest_memory=episodic_memory,  # only when its digest_search provider is configured
    semantic_memory=semantic_memory,
    procedural_memory=procedural_memory,
).build(
    PromptRequest(account_name="junwin", current_input=question,
                  conversation_id=session_id, context_name="skinny",
                  semantic_namespaces=("books", "vol_6", "vol_7", "documents")),
    PromptPolicy(total_tokens=6000, safety_margin_tokens=500,
                 procedural_tokens=300, episodic_event_tokens=800,
                 episodic_digest_tokens=400, semantic_tokens=900,
                 maximum_events=6, maximum_digests=2,
                 maximum_semantic_documents=3,
                 digest_score_threshold=0.45,
                 semantic_score_threshold=0.35,
                 semantic_item_max_chars=1800),
)
```


## Explicit episodic interfaces

The compiler's
`episodic_memory` implements `EventStore` and the active snapshot operation from
`CurationStore`; `digest_memory` is an optional, separately supplied `DigestStore`.
The same SQLite/JSONL object can supply both when it has a configured
`digest_search` provider. Omit `digest_memory` for recent-history-only use.
No agent selector or combined `EpisodicMemoryRequest`/`recall` is used.

Recent messages come from `get_recent_events`, with kind filters applied before
`maximum_events`. Complete JSON content reaches the compiler, which owns text
rendering, truncation, relevance and token budgets. The active snapshot supplies
only the current archive boundary summary as an additional required candidate;
it does not select older pre-reset boundaries. That summary uses the event
budget, independently of historical digest search. Search uses
`search_digests(account_name=..., query=..., count=...)` and the separate digest
budget. Both paths respect memory's invalidation/provenance rules.

Zero `maximum_events` or `maximum_digests` disables the corresponding read;
negative counts are rejected. No conversation ID means no recent-history read,
but explicit digest search can still run across account-owned sessions.
Compilation remains read-only. A context reset retains storage history while
removing prior events and boundary summaries from recent prompt context.

This is a breaking API/storage change requiring a fresh database. Supply
`digest_memory` explicitly if historical digest search is wanted.


## Prompt comparison CLI

The installed `galet-prompt-run` command compiles a prompt from an existing
chat database and a new request. It resolves the chat's friendly name to its stored
session ID and reads episodic history from the SQLite chat database.

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

The runner also loads the chat's stored project context, if present, from
`<storage-root>/<storage-namespace>/contexts/<account>/<context-name>.md`.
Imported skills come from the sibling `skills/<account>` directory. To try a
different named context, override the chat's value explicitly:

```bash
galet-prompt-run "What should we do next?" \
  --chat-name "Prompt Builder Work" \
  --context-name galet-memory \
  --procedural-tokens 1000
```

Use `--procedural-root /path/to/root` when the contexts and skills live
elsewhere. Use `--procedural-layout scoped` for galet-memory's global/account/
project directory layout, and `--project-name` to include a project scope.
An explicitly named context that is missing produces
an error; recall never creates a file. Set `--procedural-tokens 0` to omit
procedural content for a comparison run.

Semantic recall is opt-in. Supply one or more namespaces to query the
`embeddings-v2.sqlite` store:

```bash
galet-prompt-run \
  "What did I write about attention?" \
  --chat-name "Prompt Builder Work" \
  --namespaces vol_6 vol_7 documents
```

The comparison runner uses deliberately compact defaults:

- 8,000 total tokens with a 500-token safety margin
- 1,000 project-context and skill tokens
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

With no `--namespaces`, document semantic recall is disabled. When the
embedding database exists, the runner can still query its `digests` namespace
for archived, published session digests. A `session_digest` event in the
current chat appears as an episodic event; it does not by itself create an
archived digest candidate. Archives must be published with an embedding and
readable digest document to be recalled from other sessions. Semantic mode
defaults to
`/home/junwin/lucy_storage/data/embeddings-v2.sqlite`; use
`--embedding-db` to override it and install the optional dependency with
`python -m pip install -e ".[semantic]"`. Galet supplies the query embedding, using its
normal credential configuration or the directory passed to
`--credential-path`.

The text report includes a content-safe candidate decision table. It shows
each candidate ID, source, raw relevance score, original/final token cost, and
whether it was selected or rejected by the relevance threshold or token
budget. The JSON report exposes the same data under `metrics.candidates`.
Rejected candidates still count as retrieved and dropped in their section
metrics, making threshold tuning visible without copying document or chat
content into diagnostics.

## Dependency rule

> Applications may depend on `galet-prompt-builder`; the package must never
> depend on an application.

The package depends on the neutral interfaces from `galet-memory`. It never
opens memory databases itself and prompt compilation never mutates memory.
