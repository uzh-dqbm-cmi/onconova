"""
Match explorer: baseline patient-case count plus per–case-selector deltas.

Reuses ``patient-cases/match-count`` filtering by applying property-path removals
to a deep-copied ``caseExample`` for each selector option.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from onconova.oncology.match_count import (
    _FUNCTIONAL_AGGREGATION_PANEL_KEYS,
    interpolated_sql_for_case_example_filter,
    parse_case_example_query_param,
    patient_cases_queryset_for_case_example,
)

_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*|\[\d+\]")


def _parse_flat_property_path(path: str) -> list[tuple[str, str | int]]:
    """Parse dashboard-style paths like ``neoplasticEntities[0].morphology.code``."""
    tokens = _TOKEN_RE.findall(path.strip())
    if not tokens:
        return []
    out: list[tuple[str, str | int]] = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t.startswith("["):
            out.append(("index", int(t[1:-1])))
            i += 1
            continue
        if i + 1 < len(tokens) and tokens[i + 1].startswith("["):
            out.append(("key", t))
            out.append(("index", int(tokens[i + 1][1:-1])))
            i += 2
        else:
            out.append(("key", t))
            i += 1
    return out


def _delete_leaf_at_path(root: Any, path: str) -> bool:
    """Remove the final dict key; return True if a key was deleted."""
    steps = _parse_flat_property_path(path)
    if len(steps) < 2:
        return False
    parent_steps, last = steps[:-1], steps[-1]
    if last[0] != "key":
        return False
    leaf_key = last[1]
    if not isinstance(leaf_key, str):
        return False
    cur: Any = root
    try:
        for kind, val in parent_steps:
            if kind == "key":
                if not isinstance(cur, dict) or val not in cur:
                    return False
                cur = cur[val]
            else:
                if not isinstance(cur, list) or not isinstance(val, int):
                    return False
                if val < 0 or val >= len(cur):
                    return False
                cur = cur[val]
        if not isinstance(cur, dict):
            return False
        if leaf_key not in cur:
            return False
        del cur[leaf_key]
        return True
    except (KeyError, IndexError, TypeError, ValueError):
        return False


def _prune_empty_nested(obj: Any) -> None:
    """Drop empty dicts inside structures (mutates in place)."""
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            v = obj[k]
            _prune_empty_nested(v)
            if isinstance(v, dict) and len(v) == 0:
                del obj[k]
            elif isinstance(v, list):
                for j in range(len(v) - 1, -1, -1):
                    item = v[j]
                    if isinstance(item, dict):
                        _prune_empty_nested(item)
                        if len(item) == 0:
                            del v[j]
    elif isinstance(obj, list):
        for item in obj:
            _prune_empty_nested(item)


def _prune_empty_panel_rows(bundle: dict[str, Any]) -> None:
    """Remove empty row dicts from known panel arrays."""
    for k in list(bundle.keys()):
        if k not in _FUNCTIONAL_AGGREGATION_PANEL_KEYS:
            continue
        rows = bundle[k]
        if not isinstance(rows, list):
            continue
        bundle[k] = [r for r in rows if isinstance(r, dict) and len(r) > 0]


def remove_property_paths_from_bundle(
    bundle: dict[str, Any], property_paths: list[str]
) -> None:
    """Mutate ``bundle`` by deleting leaves and pruning empty rows (in place)."""
    ranked = sorted(
        (p for p in property_paths if isinstance(p, str) and p.strip()),
        key=lambda p: len(_parse_flat_property_path(p)),
        reverse=True,
    )
    for p in ranked:
        _delete_leaf_at_path(bundle, p)
    _prune_empty_nested(bundle)
    _prune_empty_panel_rows(bundle)


def normalize_case_example_inner(raw: dict[str, Any] | str) -> dict[str, Any]:
    """Match ``match-count`` body handling (inner partial bundle only)."""
    if isinstance(raw, str):
        if not raw.strip():
            raise ValueError(
                "caseExample must be a non-empty JSON string when sent as a string."
            )
        payload = parse_case_example_query_param(raw.strip())
    else:
        payload = raw
    if not isinstance(payload, dict):
        raise ValueError(
            "caseExample must be a JSON object or a string that parses to one."
        )
    inner = payload.get("caseExample")
    if isinstance(inner, dict):
        return copy.deepcopy(inner)
    return copy.deepcopy(payload)


def run_match_explorer(
    case_example: dict[str, Any] | str,
    case_selector_options: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Return ``patientCaseCount``, ``patientCountSql``, and ``caseSelectorOptions``
    with ``patientCaseCountIncrease`` per option.
    """
    base = normalize_case_example_inner(case_example)
    root_qs = patient_cases_queryset_for_case_example(base)
    root_n = root_qs.count()
    sql = interpolated_sql_for_case_example_filter(root_qs)

    out_opts: list[dict[str, Any]] = []
    for opt in case_selector_options:
        path = opt.get("path", "")
        if not isinstance(path, str):
            path = str(path)
        props = opt.get("properties", [])
        if not isinstance(props, list):
            props = []
        props_str = [str(x) for x in props if isinstance(x, str) and x.strip()]
        reduced = copy.deepcopy(base)
        remove_property_paths_from_bundle(reduced, props_str)
        try:
            n = patient_cases_queryset_for_case_example(reduced).count()
        except ValueError:
            n = root_n
        out_opts.append(
            {
                "path": path,
                "patientCaseCountIncrease": n - root_n,
            }
        )

    return {
        "patientCaseCount": root_n,
        "patientCountSql": sql,
        "caseSelectorOptions": out_opts,
    }
