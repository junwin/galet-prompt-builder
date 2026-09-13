from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any, Optional, Sequence

from galet_memory import (
    EpisodicMemory,
    EpisodicMemoryRequest,
    ProceduralMemory,
    ProceduralMemoryRequest,
    SemanticMemory,
    SemanticMemoryRequest,
)

from .budgets import (
    ApproximateTokenCounter,
    TokenCounter,
    truncate_to_tokens,
    validate_budgets,
)
from .candidates import PromptCandidate
from .contracts import (
    PromptBudgets,
    PromptLimits,
    PromptMessage,
    PromptRequest,
    PromptSource,
)
from .errors import MemoryRetrievalError, PromptBudgetExceededError
from .metrics import CompiledPrompt, PromptMetrics, SectionMetrics
from .relevance import DeterministicRelevanceAssessor, RelevanceAssessor
from .rendering import render_memory_messages


_SELECTION_ORDER: tuple[PromptSource, ...] = (
    "procedural",
    "episodic_event",
    "semantic",
    "episodic_digest",
)


class PromptCompiler:
    """Compile a bounded prompt using provider-neutral CoALA memory."""

    def __init__(
        self,
        *,
        procedural_memory: Optional[ProceduralMemory] = None,
        episodic_memory: Optional[EpisodicMemory] = None,
        semantic_memory: Optional[SemanticMemory] = None,
        token_counter: Optional[TokenCounter] = None,
        relevance_assessor: Optional[RelevanceAssessor] = None,
    ) -> None:
        self.procedural_memory = procedural_memory
        self.episodic_memory = episodic_memory
        self.semantic_memory = semantic_memory
        self.token_counter = token_counter or ApproximateTokenCounter()
        self.relevance_assessor = (
            relevance_assessor or DeterministicRelevanceAssessor()
        )

    def compile(
        self,
        request: PromptRequest,
        budgets: PromptBudgets,
        limits: PromptLimits,
    ) -> CompiledPrompt:
        validate_budgets(budgets, limits)
        fixed_messages = self._fixed_messages(request)
        fixed_tokens = sum(
            self._message_tokens(message, limits) for message in fixed_messages
        )
        usable_limit = budgets.total_tokens - budgets.safety_margin_tokens
        if fixed_tokens > usable_limit:
            raise PromptBudgetExceededError(
                "mandatory system instructions and current input require "
                f"{fixed_tokens} tokens, exceeding the usable limit of "
                f"{usable_limit}"
            )

        candidates = self._retrieve_candidates(request, limits)
        candidates = list(
            self.relevance_assessor.assess(
                request.semantic_query or request.current_input,
                candidates,
            )
        )
        selected, section_metrics, warnings = self._select(
            candidates,
            budgets=budgets,
            limits=limits,
            global_tokens=usable_limit - fixed_tokens,
        )
        memory_messages = render_memory_messages(selected)
        messages = (
            tuple(fixed_messages[:-1])
            + tuple(memory_messages)
            + tuple(fixed_messages[-1:])
        )
        total_used = sum(
            self._message_tokens(message, limits) for message in messages
        )
        if total_used > usable_limit:
            raise PromptBudgetExceededError(
                "compiled prompt exceeded its usable token limit"
            )

        system_messages = [
            item for item in fixed_messages if item.source == "system"
        ]
        current_messages = [
            item for item in fixed_messages if item.source == "current_input"
        ]
        metrics = PromptMetrics(
            total_limit_tokens=budgets.total_tokens,
            total_used_tokens=total_used,
            safety_margin_tokens=budgets.safety_margin_tokens,
            system=self._fixed_metrics(system_messages, limits),
            procedural=section_metrics["procedural"],
            episodic_events=section_metrics["episodic_event"],
            episodic_digests=section_metrics["episodic_digest"],
            semantic=section_metrics["semantic"],
            current_input=self._fixed_metrics(current_messages, limits),
            warnings=tuple(warnings),
        )
        return CompiledPrompt(messages=messages, metrics=metrics)

    def _fixed_messages(self, request: PromptRequest) -> list[PromptMessage]:
        messages = [
            PromptMessage("system", text.strip(), "system")
            for text in request.system_instructions
            if text and text.strip()
        ]
        messages.append(
            PromptMessage("user", request.current_input, "current_input")
        )
        return messages

    def _retrieve_candidates(
        self, request: PromptRequest, limits: PromptLimits
    ) -> list[PromptCandidate]:
        candidates: list[PromptCandidate] = []
        if request.include_procedural and self.procedural_memory:
            candidates.extend(self._recall_procedural(request, limits))
        if request.include_episodic and self.episodic_memory:
            candidates.extend(self._recall_episodic(request, limits))
        if request.include_semantic and self.semantic_memory:
            candidates.extend(self._recall_semantic(request, limits))
        return self._deduplicate_digests(candidates)

    def _recall_procedural(
        self, request: PromptRequest, limits: PromptLimits
    ) -> list[PromptCandidate]:
        if not request.context_name:
            return []
        try:
            result = self.procedural_memory.recall(
                ProceduralMemoryRequest(
                    account_name=request.account_name,
                    context_name=request.context_name,
                    create_if_missing=False,
                    include_resolved_text=True,
                    include_skills=True,
                    include_required_tools=True,
                )
            )
        except Exception as exc:
            raise MemoryRetrievalError("procedural memory recall failed") from exc
        output: list[PromptCandidate] = []
        context_text = result.resolved_text or result.text
        if context_text.strip():
            output.append(
                self._candidate(
                    "procedural:context",
                    "procedural",
                    "system",
                    f"Project context:\n{context_text}",
                    limits,
                    required=True,
                )
            )
        for order, skill in enumerate(result.skills, start=1):
            if not skill.text.strip():
                continue
            output.append(
                self._candidate(
                    f"procedural:skill:{skill.name}",
                    "procedural",
                    "system",
                    f"Skill: {skill.name}\n{skill.text}",
                    limits,
                    required=bool(skill.mandatory_tools),
                    order=order,
                )
            )
        return output

    def _recall_episodic(
        self, request: PromptRequest, limits: PromptLimits
    ) -> list[PromptCandidate]:
        try:
            result = self.episodic_memory.recall(
                EpisodicMemoryRequest(
                    account_name=request.account_name,
                    agent_name="",
                    conversation_id=request.conversation_id,
                    query=request.semantic_query or request.current_input,
                    max_events=limits.maximum_events,
                    digest_top_k=limits.maximum_digests,
                    digest_max_chars=limits.maximum_item_chars,
                    include_session_metadata=bool(request.conversation_id),
                    include_recent_history=bool(request.conversation_id),
                    include_archived_digests=request.include_digests,
                )
            )
        except Exception as exc:
            raise MemoryRetrievalError("episodic memory recall failed") from exc
        output: list[PromptCandidate] = []
        for order, event in enumerate(result.events, start=1):
            content = self._content_text(event.content)
            if not content.strip():
                continue
            role = event.role if event.role in {"user", "assistant"} else "system"
            if role == "system" and event.kind != "session_digest":
                content = f"Episodic event ({event.kind or event.role}):\n{content}"
            output.append(
                self._candidate(
                    event.event_id or f"episodic:event:{order}",
                    "episodic_event",
                    role,
                    content,
                    limits,
                    required=event.kind == "session_digest"
                    and event.metadata.get("visibility_boundary") is True,
                    order=order,
                )
            )
        if request.include_digests:
            for order, digest in enumerate(result.digests, start=1):
                content = (
                    f"Relevant session digest ({digest.session_id}):\n"
                    f"{digest.snippet}"
                )
                output.append(
                    self._candidate(
                        f"episodic:digest:{digest.session_id}:{order}",
                        "episodic_digest",
                        "system",
                        content,
                        limits,
                        relevance=digest.score,
                        order=order,
                    )
                )
        return output

    def _recall_semantic(
        self, request: PromptRequest, limits: PromptLimits
    ) -> list[PromptCandidate]:
        query = request.semantic_query or request.current_input
        if not query.strip():
            return []
        try:
            result = self.semantic_memory.recall(
                SemanticMemoryRequest(
                    account_name=request.account_name,
                    query=query,
                    namespaces=list(request.semantic_namespaces),
                    top_k=limits.maximum_semantic_documents,
                    max_chars=limits.maximum_item_chars,
                    score_threshold=0.0,
                )
            )
        except Exception as exc:
            raise MemoryRetrievalError("semantic memory recall failed") from exc
        output: list[PromptCandidate] = []
        for order, document in enumerate(result.documents, start=1):
            tags = f" | tags: {', '.join(document.tags)}" if document.tags else ""
            content = f"Relevant document: {document.title}{tags}\n{document.snippet}"
            output.append(
                self._candidate(
                    f"semantic:{document.source_id}:{order}",
                    "semantic",
                    "system",
                    content,
                    limits,
                    relevance=document.score or 0.0,
                    order=order,
                )
            )
        return output

    def _candidate(
        self,
        candidate_id: str,
        source: PromptSource,
        role: str,
        content: str,
        limits: PromptLimits,
        *,
        relevance: float = 0.0,
        required: bool = False,
        order: int = 0,
    ) -> PromptCandidate:
        truncated = len(content) > limits.maximum_item_chars
        return PromptCandidate(
            candidate_id=candidate_id,
            source=source,
            role=role,
            content=(
                content[: max(0, limits.maximum_item_chars - 1)].rstrip()
                + "…"
                if truncated
                else content
            ),
            relevance=relevance,
            required=required,
            order=order,
            truncated=truncated,
        )

    def _deduplicate_digests(
        self, candidates: Sequence[PromptCandidate]
    ) -> list[PromptCandidate]:
        active_digest_fingerprints = {
            self._fingerprint(item.content)
            for item in candidates
            if item.source == "episodic_event" and item.required
        }
        seen = set(active_digest_fingerprints)
        output: list[PromptCandidate] = []
        for item in candidates:
            if item.source != "episodic_digest":
                output.append(item)
                continue
            fingerprint = self._fingerprint(
                item.content.split("\n", 1)[-1]
            )
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            output.append(item)
        return output

    def _select(
        self,
        candidates: Sequence[PromptCandidate],
        *,
        budgets: PromptBudgets,
        limits: PromptLimits,
        global_tokens: int,
    ) -> tuple[
        list[PromptCandidate], dict[PromptSource, SectionMetrics], list[str]
    ]:
        section_budgets: dict[PromptSource, int] = {
            "procedural": budgets.procedural_tokens,
            "episodic_event": budgets.episodic_event_tokens,
            "episodic_digest": budgets.episodic_digest_tokens,
            "semantic": budgets.semantic_tokens,
        }
        selected: list[PromptCandidate] = []
        metrics: dict[PromptSource, SectionMetrics] = {}
        warnings: list[str] = []
        remaining_global = global_tokens
        for source in _SELECTION_ORDER:
            items = [item for item in candidates if item.source == source]
            ranked = sorted(
                items,
                key=lambda item: (
                    item.required,
                    item.relevance,
                    item.order,
                ),
                reverse=True,
            )
            remaining_section = min(
                section_budgets[source], remaining_global
            )
            chosen: list[PromptCandidate] = []
            used = 0
            truncated = 0
            for item in ranked:
                cost = self._candidate_tokens(item, limits)
                if cost <= remaining_section:
                    chosen.append(item)
                    used += cost
                    remaining_section -= cost
                    if item.truncated:
                        truncated += 1
                    continue
                content_budget = remaining_section - limits.message_overhead_tokens
                shortened = truncate_to_tokens(
                    item.content,
                    content_budget,
                    self.token_counter,
                )
                if shortened:
                    shortened_item = replace(
                        item, content=shortened, truncated=True
                    )
                    shortened_cost = self._candidate_tokens(
                        shortened_item, limits
                    )
                    chosen.append(shortened_item)
                    used += shortened_cost
                    remaining_section -= shortened_cost
                    truncated += 1
                break
            selected.extend(chosen)
            remaining_global -= used
            dropped = len(items) - len(chosen)
            if dropped or truncated:
                warnings.append(
                    f"{source}: dropped={dropped}, truncated={truncated}"
                )
            metrics[source] = SectionMetrics(
                budget_tokens=section_budgets[source],
                used_tokens=used,
                retrieved_items=len(items),
                selected_items=len(chosen),
                dropped_items=dropped,
                truncated_items=truncated,
            )
        return selected, metrics, warnings

    def _message_tokens(
        self, message: PromptMessage, limits: PromptLimits
    ) -> int:
        return (
            self.token_counter.count(message.content)
            + limits.message_overhead_tokens
        )

    def _candidate_tokens(
        self, candidate: PromptCandidate, limits: PromptLimits
    ) -> int:
        return (
            self.token_counter.count(candidate.content)
            + limits.message_overhead_tokens
        )

    def _fixed_metrics(
        self, messages: Sequence[PromptMessage], limits: PromptLimits
    ) -> SectionMetrics:
        used = sum(self._message_tokens(item, limits) for item in messages)
        return SectionMetrics(
            budget_tokens=used,
            used_tokens=used,
            retrieved_items=len(messages),
            selected_items=len(messages),
        )

    @staticmethod
    def _content_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False, sort_keys=True)

    @staticmethod
    def _fingerprint(content: str) -> str:
        normalized = " ".join(content.casefold().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


__all__ = ["PromptCompiler"]
