from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

from feature_layer.indicator_registry import active_feature_spec, fields_for_indicator
from feature_layer.specs import FeatureField, FeatureSpec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_INPUT_PATH = PROJECT_ROOT / "config" / "features" / "model_input.json"
ITEM_TYPES = ("group", "comment", "feature")
INPUT_STREAMS = ("market_sequence", "decision_context", "runtime_state")
MISSING_POLICIES = ("zero",)
_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")


def model_input_blueprint_payload(
    path: str | Path = DEFAULT_MODEL_INPUT_PATH,
    *,
    spec: FeatureSpec | None = None,
) -> dict[str, Any]:
    active_spec = spec or active_feature_spec()
    blueprint = load_model_input_blueprint(path, spec=active_spec)
    validation = validate_model_input_blueprint(blueprint, spec=active_spec)
    compiled = (
        compile_model_input_blueprint(blueprint, spec=active_spec)
        if validation["valid"]
        else _empty_compiled(blueprint, active_spec)
    )
    return {
        "path": _display_path(Path(path)),
        "blueprint": blueprint,
        "default_blueprint": default_model_input_blueprint(active_spec),
        "catalog": model_input_catalog(active_spec),
        "available_frequencies": _available_frequencies(active_spec),
        "input_streams": list(INPUT_STREAMS),
        "missing_policies": list(MISSING_POLICIES),
        "validation": validation,
        "compiled": compiled,
    }


def load_model_input_blueprint(
    path: str | Path = DEFAULT_MODEL_INPUT_PATH,
    *,
    spec: FeatureSpec | None = None,
) -> dict[str, Any]:
    active_spec = spec or active_feature_spec()
    config_path = Path(path)
    if config_path.exists():
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    else:
        raw = default_model_input_blueprint(active_spec)
    normalized = _normalize_blueprint(raw)
    return _with_hashes(normalized)


def save_model_input_blueprint(
    payload: dict[str, Any],
    path: str | Path = DEFAULT_MODEL_INPUT_PATH,
    *,
    spec: FeatureSpec | None = None,
) -> dict[str, Any]:
    active_spec = spec or active_feature_spec()
    config_path = Path(path)
    incoming = payload.get("blueprint") if isinstance(payload.get("blueprint"), dict) else payload
    normalized = _normalize_blueprint(incoming)
    validate_model_input_blueprint(normalized, spec=active_spec, raise_on_error=True)

    previous = None
    if config_path.exists():
        try:
            previous = load_model_input_blueprint(config_path, spec=active_spec)
        except Exception:
            previous = None
    previous_schema_hash = previous.get("schema_hash") if previous else None
    next_schema_hash = _schema_hash(normalized)
    normalized["document_version"] = int(previous.get("document_version", 0) if previous else 0) + 1
    previous_schema_version = int(previous.get("schema_version", 0) if previous else 0)
    normalized["schema_version"] = (
        previous_schema_version
        if previous_schema_hash == next_schema_hash and previous_schema_version > 0
        else previous_schema_version + 1
    )
    document = _with_hashes(normalized)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return model_input_blueprint_payload(config_path, spec=active_spec)


def default_model_input_blueprint(spec: FeatureSpec | None = None) -> dict[str, Any]:
    active_spec = spec or active_feature_spec()
    frequencies = list(_available_frequencies(active_spec))
    items: list[dict[str, Any]] = []
    last_group = None
    for field in active_spec.market_fields:
        if field.group != last_group:
            last_group = field.group
            items.append(_group_item(f"market_{last_group}", _group_label(last_group)))
        items.append(_feature_item(field.name, "market_sequence", frequencies))

    items.append(_group_item("decision_context", "Decision Context"))
    items.extend(_feature_item(field.name, "decision_context") for field in active_spec.context_fields)
    items.append(_group_item("market_constraints", "Market Constraints"))
    items.extend(_feature_item(field.name, "decision_context") for field in active_spec.constraint_fields)
    items.append(_group_item("runtime_portfolio", "Runtime Portfolio"))
    items.extend(_feature_item(field.name, "runtime_state") for field in active_spec.portfolio_fields)
    items.append(_group_item("environment_state", "Environment State"))
    items.extend(_feature_item(field.name, "runtime_state") for field in active_spec.environment_fields)
    return _with_hashes(
        {
            "document_version": 1,
            "schema_version": 1,
            "items": items,
        }
    )


def validate_model_input_blueprint(
    blueprint: dict[str, Any],
    *,
    spec: FeatureSpec | None = None,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    active_spec = spec or active_feature_spec()
    catalog = {item["name"]: item for item in model_input_catalog(active_spec)}
    allowed_frequencies = set(_available_frequencies(active_spec))
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    seen_item_ids: set[str] = set()
    seen_channels: set[tuple[str, str, str]] = set()
    market_counts = {freq: 0 for freq in allowed_frequencies}

    for index, item in enumerate(blueprint.get("items", [])):
        item_id = str(item.get("id") or "")
        item_type = str(item.get("type") or "")
        if not _ID_PATTERN.fullmatch(item_id):
            errors.append(_issue(index, item_id, "invalid_id", "Item ID is missing or invalid."))
        elif item_id in seen_item_ids:
            errors.append(_issue(index, item_id, "duplicate_id", "Item ID must be unique."))
        seen_item_ids.add(item_id)
        if item_type not in ITEM_TYPES:
            errors.append(_issue(index, item_id, "invalid_type", f"Unsupported item type: {item_type}"))
            continue
        if item_type != "feature":
            continue

        name = str(item.get("name") or "")
        stream = str(item.get("stream") or "")
        catalog_item = catalog.get(name)
        if catalog_item is None:
            errors.append(_issue(index, item_id, "unknown_feature", f"Unknown feature: {name}"))
            continue
        if stream not in INPUT_STREAMS:
            errors.append(_issue(index, item_id, "invalid_stream", f"Unsupported input stream: {stream}"))
            continue
        if stream != catalog_item["stream"]:
            errors.append(
                _issue(
                    index,
                    item_id,
                    "wrong_stream",
                    f"{name} belongs to {catalog_item['stream']}, not {stream}.",
                )
            )
        if str(item.get("missing_policy") or "zero") not in MISSING_POLICIES:
            errors.append(_issue(index, item_id, "missing_policy", "Unsupported missing-value policy."))
        if not bool(item.get("enabled", True)):
            continue

        frequencies = [str(value) for value in item.get("frequencies", [])]
        if stream == "market_sequence":
            if not frequencies:
                errors.append(_issue(index, item_id, "missing_frequency", "Market features need at least one frequency."))
            unsupported = sorted(set(frequencies).difference(allowed_frequencies))
            if unsupported:
                errors.append(_issue(index, item_id, "invalid_frequency", f"Unsupported frequencies: {', '.join(unsupported)}"))
            unavailable = sorted(set(frequencies).difference(catalog_item["available_frequencies"]))
            if unavailable:
                errors.append(_issue(index, item_id, "unavailable_frequency", f"{name} is unavailable for: {', '.join(unavailable)}"))
            for freq in frequencies:
                key = (stream, freq, name)
                if key in seen_channels:
                    errors.append(_issue(index, item_id, "duplicate_channel", f"Duplicate channel: {freq} / {name}"))
                seen_channels.add(key)
                if freq in market_counts:
                    market_counts[freq] += 1
        else:
            if frequencies:
                warnings.append(_issue(index, item_id, "ignored_frequency", "Non-sequence frequency selection is ignored."))
            key = (stream, "", name)
            if key in seen_channels:
                errors.append(_issue(index, item_id, "duplicate_channel", f"Duplicate channel: {stream} / {name}"))
            seen_channels.add(key)

    for freq, count in sorted(market_counts.items()):
        if count == 0:
            errors.append(_issue(None, freq, "empty_frequency", f"No market channels selected for {freq}."))

    result = {"valid": not errors, "errors": errors, "warnings": warnings}
    if errors and raise_on_error:
        raise ValueError("Model input blueprint validation failed: " + "; ".join(item["message"] for item in errors))
    return result


def compile_model_input_blueprint(
    blueprint: dict[str, Any] | None = None,
    *,
    spec: FeatureSpec | None = None,
) -> dict[str, Any]:
    active_spec = spec or active_feature_spec()
    document = blueprint or load_model_input_blueprint(spec=active_spec)
    validate_model_input_blueprint(document, spec=active_spec, raise_on_error=True)
    frequencies = _available_frequencies(active_spec)
    channels_by_frequency = {freq: [] for freq in frequencies}
    decision_context: list[str] = []
    runtime_state: list[str] = []
    for item in document.get("items", []):
        if item.get("type") != "feature" or not bool(item.get("enabled", True)):
            continue
        stream = item["stream"]
        name = item["name"]
        if stream == "market_sequence":
            for freq in item.get("frequencies", []):
                if freq in channels_by_frequency:
                    channels_by_frequency[freq].append(name)
        elif stream == "decision_context":
            decision_context.append(name)
        else:
            runtime_state.append(name)
    return {
        "schema_version": int(document.get("schema_version", 1)),
        "schema_hash": _schema_hash(document),
        "channels_by_frequency": channels_by_frequency,
        "decision_context": decision_context,
        "runtime_state": runtime_state,
        "shapes": {
            freq: [int(active_spec.sequence_windows.get(freq, 0)), len(names)]
            for freq, names in channels_by_frequency.items()
        },
        "decision_context_shape": [len(decision_context)],
        "runtime_state_shape": [len(runtime_state)],
    }


def model_input_catalog(spec: FeatureSpec | None = None) -> list[dict[str, Any]]:
    active_spec = spec or active_feature_spec()
    frequencies = list(_available_frequencies(active_spec))
    technical_fields: dict[str, tuple[str, ...]] = {}
    for indicator in active_spec.indicators:
        for field in fields_for_indicator(indicator):
            technical_fields[field.name] = indicator.frequencies

    result: list[dict[str, Any]] = []
    for field in active_spec.market_fields:
        result.append(
            _catalog_item(
                field,
                stream="market_sequence",
                category="technical" if field.name in technical_fields else "non_technical",
                frequencies=list(technical_fields.get(field.name, frequencies)),
            )
        )
    result.extend(_catalog_item(field, stream="decision_context", category="non_technical") for field in active_spec.context_fields)
    result.extend(_catalog_item(field, stream="decision_context", category="non_technical") for field in active_spec.constraint_fields)
    result.extend(_catalog_item(field, stream="runtime_state", category="non_technical") for field in active_spec.portfolio_fields)
    result.extend(_catalog_item(field, stream="runtime_state", category="non_technical") for field in active_spec.environment_fields)
    return result


def _catalog_item(
    field: FeatureField,
    *,
    stream: str,
    category: str,
    frequencies: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "name": field.name,
        "stream": stream,
        "group": field.group,
        "category": category,
        "description": field.description,
        "clip": list(field.clip) if field.clip is not None else None,
        "available_frequencies": frequencies or [],
    }


def _normalize_blueprint(raw: dict[str, Any]) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    for index, source in enumerate(raw.get("items", [])):
        item_type = str(source.get("type") or "feature")
        item = {
            "id": str(source.get("id") or f"item_{index + 1}"),
            "type": item_type,
        }
        if item_type == "group":
            item["label"] = str(source.get("label") or "Group")
        elif item_type == "comment":
            item["text"] = str(source.get("text") or "")
        else:
            item.update(
                {
                    "name": str(source.get("name") or ""),
                    "stream": str(source.get("stream") or "market_sequence"),
                    "frequencies": list(dict.fromkeys(str(value) for value in source.get("frequencies", []))),
                    "missing_policy": str(source.get("missing_policy") or "zero"),
                    "enabled": bool(source.get("enabled", True)),
                }
            )
        items.append(item)
    return {
        "document_version": max(1, int(raw.get("document_version", 1))),
        "schema_version": max(1, int(raw.get("schema_version", 1))),
        "items": items,
    }


def _with_hashes(document: dict[str, Any]) -> dict[str, Any]:
    result = {
        "document_version": int(document.get("document_version", 1)),
        "schema_version": int(document.get("schema_version", 1)),
        "items": document.get("items", []),
    }
    result["schema_hash"] = _schema_hash(result)
    result["document_hash"] = _hash_json(result)
    return result


def _schema_hash(document: dict[str, Any]) -> str:
    schema_items = []
    for item in document.get("items", []):
        if item.get("type") != "feature" or not bool(item.get("enabled", True)):
            continue
        schema_items.append(
            {
                "name": item.get("name"),
                "stream": item.get("stream"),
                "frequencies": item.get("frequencies", []),
                "missing_policy": item.get("missing_policy", "zero"),
            }
        )
    return _hash_json({"features": schema_items})


def _hash_json(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _available_frequencies(spec: FeatureSpec) -> tuple[str, ...]:
    return (spec.base_frequency, *spec.derived_frequencies)


def _empty_compiled(document: dict[str, Any], spec: FeatureSpec) -> dict[str, Any]:
    frequencies = _available_frequencies(spec)
    return {
        "schema_version": int(document.get("schema_version", 1)),
        "schema_hash": _schema_hash(document),
        "channels_by_frequency": {freq: [] for freq in frequencies},
        "decision_context": [],
        "runtime_state": [],
        "shapes": {freq: [int(spec.sequence_windows.get(freq, 0)), 0] for freq in frequencies},
        "decision_context_shape": [0],
        "runtime_state_shape": [0],
    }


def _group_item(suffix: str, label: str) -> dict[str, Any]:
    return {"id": f"group_{suffix}", "type": "group", "label": label}


def _feature_item(name: str, stream: str, frequencies: list[str] | None = None) -> dict[str, Any]:
    return {
        "id": f"feature_{stream}_{name}",
        "type": "feature",
        "name": name,
        "stream": stream,
        "frequencies": frequencies or [],
        "missing_policy": "zero",
        "enabled": True,
    }


def _group_label(group: str) -> str:
    return str(group or "Features").replace("_", " ").title()


def _issue(index: int | None, item_id: str, code: str, message: str) -> dict[str, Any]:
    return {"index": index, "item_id": item_id, "code": code, "message": message}


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path)
