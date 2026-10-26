"""Explicit prompt policy and a reference to galet-memory's shared config."""
from dataclasses import dataclass, field, fields
import json
import math
import os
from pathlib import Path
from collections.abc import Mapping

from .budgets import validate_budgets
from .errors import PromptConfigurationError
from .policy import PromptPolicy


@dataclass(frozen=True)
class PromptBuilderSettings:
    memory_config_path: Path | None = None
    policy: PromptPolicy = field(default_factory=PromptPolicy)


def validate_policy(policy):
    defaults = PromptPolicy()
    for item in fields(PromptPolicy):
        value = getattr(policy, item.name)
        expected = getattr(defaults, item.name)
        if value is None and item.name.endswith('_max_chars'):
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise PromptConfigurationError(f'invalid policy field: {item.name}')
        if not isinstance(expected, float) and not isinstance(value, int):
            raise PromptConfigurationError(f'policy field must be an integer: {item.name}')
        if item.name.endswith('_threshold') and not 0 <= value <= 1:
            raise PromptConfigurationError(f'policy threshold must be between zero and one: {item.name}')
    validate_budgets(policy.budgets(), policy.limits())
    return policy


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PromptConfigurationError(f'duplicate prompt configuration key: {key}')
        result[key] = value
    return result


def load_prompt_config(config=None, *, path=None, base_dir=None):
    """Object > explicit file > GALET_PROMPT_BUILDER_CONFIG_PATH; no merging.

    Without a selected source, use default policy and let galet-memory select
    GALET_MEMORY_CONFIG_PATH. Relative references are anchored to the containing
    file, or to an explicit base_dir for a configuration object.
    """
    base = Path(base_dir).expanduser().resolve() if base_dir is not None else None
    if config is not None:
        if not isinstance(config, Mapping):
            raise PromptConfigurationError('prompt configuration must be an object')
        raw = dict(config)
    else:
        selected = path if path is not None else os.environ.get('GALET_PROMPT_BUILDER_CONFIG_PATH')
        if selected is None:
            return PromptBuilderSettings()
        file = Path(selected).expanduser().resolve()
        try:
            raw = json.loads(file.read_text(encoding='utf-8'), object_pairs_hook=_unique_object)
        except (OSError, ValueError) as exc:
            raise PromptConfigurationError(f'cannot load prompt configuration: {file}') from exc
        base = file.parent
    if not isinstance(raw, dict):
        raise PromptConfigurationError('prompt configuration must be an object')
    if set(raw) - {'config_version', 'memory_config_path', 'policy'}:
        raise PromptConfigurationError('unknown prompt configuration fields')
    if type(raw.get('config_version')) is not int or raw['config_version'] != 1:
        raise PromptConfigurationError('config_version must be 1')
    values = raw.get('policy', {})
    if not isinstance(values, dict) or set(values) - {f.name for f in fields(PromptPolicy)}:
        raise PromptConfigurationError('invalid prompt policy fields')
    policy = validate_policy(PromptPolicy(**values))
    reference = raw.get('memory_config_path')
    memory_path = None
    if reference is not None:
        if not isinstance(reference, str) or not reference.strip():
            raise PromptConfigurationError('memory_config_path must be a nonempty path')
        memory_path = Path(reference).expanduser()
        if not memory_path.is_absolute():
            if base is None:
                raise PromptConfigurationError('relative memory_config_path requires base_dir')
            memory_path = base / memory_path
        memory_path = memory_path.resolve()
    return PromptBuilderSettings(memory_path, policy)
