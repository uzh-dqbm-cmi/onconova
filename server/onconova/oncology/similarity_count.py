"""
Count PatientCase rows matching a case-example partial bundle (panel JSON from aitb-dashboard).

The payload shape matches the aitb-dashboard ``caseExample`` object (panel bundle; legacy
``functional_aggregated_data`` wrapper is still accepted in ``_normalize_bundle``):
camelCase panel keys (e.g. ``neoplasticEntities``) mapping to lists of row dicts
built from Onconova export slices (coded concepts as ``{ "code": "..." }``).

Panel keys are kept in sync with aitb-dashboard ``casePanelColumns`` /
``BUNDLE_ARRAY_KEYS_ORDER`` (Cancer Characterization, Genomics, Therapy, Profile,
Clinical Observations and every sub-panel ``key`` there).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterable

logger = logging.getLogger(__name__)

from django.core.exceptions import FieldDoesNotExist
from django.db import connections, models
from django.db.models import Exists, OuterRef, Q
from django.utils.encoding import force_str

from onconova.core.utils import camel_to_snake
from onconova.oncology import models as orm
from onconova.oncology.models.performance_status import (
    ECOG_INTEPRETATION,
    KARNOFSKY_INTEPRETATION,
)
from onconova.terminology.models import CodedConcept


def _coded_code(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        return s or None
    if isinstance(value, dict):
        c = value.get("code")
        if c is None:
            return None
        s = str(c).strip()
        return s or None
    return None


def _normalize_bundle(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("functional aggregated payload must be a JSON object")
    inner = payload.get("caseExample")
    if isinstance(inner, dict):
        return inner
    inner = payload.get("functional_aggregated_data")
    if isinstance(inner, dict):
        return inner
    return payload


def _is_coded_concept_relation_field(model: type[models.Model], field_name: str) -> bool:
    try:
        f = model._meta.get_field(field_name)
    except FieldDoesNotExist:
        return False
    if isinstance(f, models.ForeignKey):
        return bool(
            f.related_model is not None and issubclass(f.related_model, CodedConcept)
        )
    if isinstance(f, models.ManyToManyField):
        rel_model = f.remote_field.model
        return bool(rel_model is not None and issubclass(rel_model, CodedConcept))
    return False


def _iter_parent_linked_children(
    parent: type[models.Model],
) -> Iterable[tuple[str, type[models.Model]]]:
    for cls in parent.__subclasses__():
        if not issubclass(cls, parent):
            continue
        for f in cls._meta.local_fields:
            if not getattr(f, "parent_link", False):
                continue
            if getattr(f.remote_field, "model", None) is not parent:
                continue
            rel = f.remote_field.related_name
            if rel:
                yield rel, cls
            break


def _staging_row_q(row: dict[str, Any]) -> Q | None:
    pairs = [(camel_to_snake(k), _coded_code(v)) for k, v in row.items()]
    pairs = [(sn, c) for sn, c in pairs if c]
    if not pairs:
        return None

    matching_kwarg_lists: list[dict[str, str]] = []
    for rel, cls in _iter_parent_linked_children(orm.Staging):
        kwargs: dict[str, str] = {}
        ok = True
        for snake, code in pairs:
            if not _is_coded_concept_relation_field(cls, snake):
                ok = False
                break
            kwargs[f"{rel}__{snake}__code"] = code
        if ok and kwargs:
            matching_kwarg_lists.append(kwargs)

    if not matching_kwarg_lists:
        return Q(pk__in=[])

    if len(matching_kwarg_lists) == 1:
        return Q(
            Exists(
                orm.Staging.objects.filter(case_id=OuterRef("pk"), **matching_kwarg_lists[0])
            )
        )

    combined = Q()
    for mk in matching_kwarg_lists:
        combined |= Q(Exists(orm.Staging.objects.filter(case_id=OuterRef("pk"), **mk)))
    return combined


def _genomic_signature_row_q(row: dict[str, Any]) -> Q | None:
    pairs = [(camel_to_snake(k), _coded_code(v)) for k, v in row.items()]
    pairs = [(sn, c) for sn, c in pairs if c]
    if not pairs:
        return None

    matching_kwarg_lists: list[dict[str, str]] = []
    for rel, cls in _iter_parent_linked_children(orm.GenomicSignature):
        kwargs: dict[str, str] = {}
        ok = True
        for snake, code in pairs:
            if not _is_coded_concept_relation_field(cls, snake):
                ok = False
                break
            kwargs[f"{rel}__{snake}__code"] = code
        if ok and kwargs:
            matching_kwarg_lists.append(kwargs)

    if not matching_kwarg_lists:
        return Q(pk__in=[])

    if len(matching_kwarg_lists) == 1:
        return Q(
            Exists(
                orm.GenomicSignature.objects.filter(
                    case_id=OuterRef("pk"), **matching_kwarg_lists[0]
                )
            )
        )

    combined = Q()
    for mk in matching_kwarg_lists:
        combined |= Q(
            Exists(orm.GenomicSignature.objects.filter(case_id=OuterRef("pk"), **mk))
        )
    return combined


def _tumor_board_row_q(row: dict[str, Any]) -> Q | None:
    kwargs: dict[str, Any] = {}
    for json_key, val in row.items():
        code = _coded_code(val)
        if not code:
            continue
        snake = camel_to_snake(json_key)
        if _is_coded_concept_relation_field(orm.TumorBoard, snake):
            kwargs[f"{snake}__code"] = code

    if kwargs:
        return Q(Exists(orm.TumorBoard.objects.filter(case_id=OuterRef("pk"), **kwargs)))

    matching_kwarg_lists: list[dict[str, str]] = []
    for rel, cls in _iter_parent_linked_children(orm.TumorBoard):
        ckwargs: dict[str, str] = {}
        ok = True
        for json_key, val in row.items():
            code = _coded_code(val)
            if not code:
                continue
            snake = camel_to_snake(json_key)
            if not _is_coded_concept_relation_field(cls, snake):
                ok = False
                break
            ckwargs[f"{rel}__{snake}__code"] = code
        if ok and ckwargs:
            matching_kwarg_lists.append(ckwargs)

    if not matching_kwarg_lists:
        return None

    if len(matching_kwarg_lists) == 1:
        return Q(
            Exists(
                orm.TumorBoard.objects.filter(case_id=OuterRef("pk"), **matching_kwarg_lists[0])
            )
        )

    combined = Q()
    for mk in matching_kwarg_lists:
        combined |= Q(Exists(orm.TumorBoard.objects.filter(case_id=OuterRef("pk"), **mk)))
    return combined


def _neoplastic_row_q(row: dict[str, Any]) -> Q | None:
    kwargs: dict[str, Any] = {"case_id": OuterRef("pk")}
    rel_raw = row.get("relationship")
    rel_code = _coded_code(rel_raw) if isinstance(rel_raw, dict) else _coded_code(rel_raw)
    if rel_code:
        kwargs["relationship"] = rel_code
    topo = _coded_code(row.get("topography"))
    if topo:
        kwargs["topography__code"] = topo
    morph = _coded_code(row.get("morphology"))
    if morph:
        kwargs["morphology__code"] = morph
    tg = _coded_code(row.get("topographyGroup")) or _coded_code(row.get("topography_group"))
    if tg:
        kwargs["topography_group__code"] = tg

    if len(kwargs) == 1:
        return None
    return Q(Exists(orm.NeoplasticEntity.objects.filter(**kwargs)))


def _gene_panel_string_value(val: Any) -> str | None:
    """Resolve gene panel from export JSON (CharField on GenomicVariant, not a CodedConcept FK)."""
    if val is None:
        return None
    c = _coded_code(val)
    if c:
        return c
    if isinstance(val, str):
        s = val.strip()
        return s or None
    if isinstance(val, dict):
        d = val.get("display")
        if d is not None:
            s = str(d).strip()
            if s:
                return s
    return None


def _cytogenetic_location_filter_value(val: Any) -> str | None:
    """Cytogenetic location is a string in the API; bundles may send {code} only."""
    if val is None:
        return None
    if isinstance(val, str):
        s = val.strip()
        return s or None
    return _coded_code(val)


def _ecog_interpretation_code_to_score(code: str) -> int | None:
    for score, mapped in ECOG_INTEPRETATION.items():
        if mapped == code:
            return score
    return None


def _karnofsky_interpretation_code_to_score(code: str) -> int | None:
    for score, mapped in KARNOFSKY_INTEPRETATION.items():
        if mapped == code:
            return score
    return None


def _performance_status_row_q(row: dict[str, Any]) -> Q | None:
    """
    Build an EXISTS filter for PerformanceStatus rows.

    Export JSON uses ``ecogInterpretation`` / ``karnofskyInterpretation`` coded concepts,
    but the ORM model exposes those as ``SubqueryObjectProperty`` (derived from numeric
    scores), not real ``ForeignKey`` columns — so ``_flat_model_row_q`` never matched them.
    """
    kwargs: dict[str, Any] = {}
    for json_key, val in row.items():
        snake = camel_to_snake(json_key)
        if snake == "ecog_interpretation":
            code = _coded_code(val)
            if code:
                score = _ecog_interpretation_code_to_score(code)
                if score is None:
                    return Q(pk__in=[])
                kwargs["ecog_score"] = score
            continue
        if snake == "karnofsky_interpretation":
            code = _coded_code(val)
            if code:
                score = _karnofsky_interpretation_code_to_score(code)
                if score is None:
                    return Q(pk__in=[])
                kwargs["karnofsky_score"] = score
            continue
        if snake == "ecog_score" and val is not None:
            try:
                kwargs["ecog_score"] = int(val)
            except (TypeError, ValueError):
                pass
            continue
        if snake == "karnofsky_score" and val is not None:
            try:
                kwargs["karnofsky_score"] = int(val)
            except (TypeError, ValueError):
                pass
            continue
        code = _coded_code(val)
        if not code:
            continue
        if _is_coded_concept_relation_field(orm.PerformanceStatus, snake):
            kwargs[f"{snake}__code"] = code
    if not kwargs:
        return None
    return Q(Exists(orm.PerformanceStatus.objects.filter(case_id=OuterRef("pk"), **kwargs)))


def _genomic_variant_row_q(row: dict[str, Any]) -> Q | None:
    """
    Build an EXISTS filter for GenomicVariant rows.

    ``_flat_model_row_q`` only considers CodedConcept FK/M2M columns. GenomicVariant also
    stores ``gene_panel`` as a plain CharField, ``hgvsVersion`` is API-only (not a DB
    filter), and ``cytogenetic_location`` is a queryable ``AnnotationProperty`` that must
    be filtered by lookup on that property.
    """
    kwargs: dict[str, Any] = {}
    for json_key, val in row.items():
        snake = camel_to_snake(json_key)
        if snake == "gene_panel":
            gp = _gene_panel_string_value(val)
            if gp:
                kwargs["gene_panel__iexact"] = gp
            continue
        if snake == "hgvs_version":
            # Serialized HGVS nomenclature version (schema default); not a persisted column.
            continue
        if snake == "cytogenetic_location":
            loc = _cytogenetic_location_filter_value(val)
            if loc:
                kwargs["cytogenetic_location__icontains"] = loc
            continue
        code = _coded_code(val)
        if not code:
            continue
        if _is_coded_concept_relation_field(orm.GenomicVariant, snake):
            kwargs[f"{snake}__code"] = code
    if not kwargs:
        return None
    return Q(Exists(orm.GenomicVariant.objects.filter(case_id=OuterRef("pk"), **kwargs)))


def _flat_model_row_q(model: type[models.Model], row: dict[str, Any]) -> Q | None:
    kwargs: dict[str, str] = {}
    for json_key, val in row.items():
        code = _coded_code(val)
        if not code:
            continue
        snake = camel_to_snake(json_key)
        if _is_coded_concept_relation_field(model, snake):
            kwargs[f"{snake}__code"] = code
    if not kwargs:
        return None
    return Q(Exists(model.objects.filter(case_id=OuterRef("pk"), **kwargs)))


# Flat ORM targets keyed by aitb-dashboard panel ``key`` (see ``onconovaCasePanels``).
_PANEL_FLAT_MODELS: dict[str, type[models.Model]] = {
    "tumorMarkers": orm.TumorMarker,
    "riskAssessments": orm.RiskAssessment,
    "therapyLines": orm.TherapyLine,
    "systemicTherapies": orm.SystemicTherapy,
    "surgeries": orm.Surgery,
    "radiotherapies": orm.Radiotherapy,
    "adverseEvents": orm.AdverseEvent,
    "treatmentResponses": orm.TreatmentResponse,
    "comorbidities": orm.ComorbiditiesAssessment,
    "vitals": orm.Vitals,
    "lifestyles": orm.Lifestyle,
    "familyHistory": orm.FamilyHistory,
}

# Every panel key the dashboard can emit under ``caseExample`` (bundle root); must match
# ``BUNDLE_ARRAY_KEYS_ORDER`` in aitb-dashboard ``PatientCaseView.vue``.
_FUNCTIONAL_AGGREGATION_PANEL_KEYS: frozenset[str] = frozenset(
    {
        # Cancer Characterization
        "neoplasticEntities",
        "stagings",
        "tumorMarkers",
        "riskAssessments",
        # Genomics
        "genomicVariants",
        "genomicSignatures",
        # Therapy
        "therapyLines",
        "systemicTherapies",
        "surgeries",
        "radiotherapies",
        # Profile
        "lifestyles",
        "familyHistory",
        "comorbidities",
        "vitals",
        # Clinical Observations
        "tumorBoards",
        "adverseEvents",
        "treatmentResponses",
        "performanceStatus",
    }
)

_POLYMORPHIC_PANEL_KEYS: frozenset[str] = frozenset(
    {"neoplasticEntities", "stagings", "genomicSignatures", "tumorBoards"}
)

# Panels with a dedicated row handler in ``_panel_row_q`` (not plain ``_flat_model_row_q``).
_SPECIAL_PANEL_ROW_KEYS: frozenset[str] = frozenset(
    {"genomicVariants", "performanceStatus"}
)


def _assert_panel_registry_matches_dashboard() -> None:
    """Invariant: every dashboard panel key maps to a handler (tests / CI)."""
    handled = (
        frozenset(_PANEL_FLAT_MODELS)
        | _POLYMORPHIC_PANEL_KEYS
        | _SPECIAL_PANEL_ROW_KEYS
    )
    if handled != _FUNCTIONAL_AGGREGATION_PANEL_KEYS:
        missing = sorted(_FUNCTIONAL_AGGREGATION_PANEL_KEYS - handled)
        extra = sorted(handled - _FUNCTIONAL_AGGREGATION_PANEL_KEYS)
        raise RuntimeError(
            "similarity count panel registry drift: "
            f"missing_handlers={missing!r} extra_handlers={extra!r}"
        )


_assert_panel_registry_matches_dashboard()


def _sql_with_params_interpolated_for_log(db_alias: str, sql: str, params: Any) -> str:
    """Return SQL with bound parameters inlined (PostgreSQL only); for logs, not execution."""
    conn = connections[db_alias]
    if conn.vendor != "postgresql":
        return f"{sql}\n-- params (raw, not interpolated on {conn.vendor!r}): {params!r}"
    from django.db.backends.postgresql.psycopg_any import mogrify

    try:
        raw = mogrify(sql, () if params is None else params, conn)
    except Exception as exc:  # pragma: no cover - defensive
        return f"{sql}\n-- params: {params!r}\n-- mogrify failed: {exc!r}"
    return force_str(raw)


def _panel_row_q(panel_key: str, row: dict[str, Any]) -> Q | None:
    if panel_key == "neoplasticEntities":
        return _neoplastic_row_q(row)
    if panel_key == "stagings":
        return _staging_row_q(row)
    if panel_key == "genomicSignatures":
        return _genomic_signature_row_q(row)
    if panel_key == "tumorBoards":
        return _tumor_board_row_q(row)
    if panel_key == "genomicVariants":
        return _genomic_variant_row_q(row)
    if panel_key == "performanceStatus":
        return _performance_status_row_q(row)
    model = _PANEL_FLAT_MODELS.get(panel_key)
    if model:
        return _flat_model_row_q(model, row)
    return None


def patient_cases_queryset_for_functional_aggregated_data(payload: Any) -> models.QuerySet:
    """Return the filtered PatientCase queryset for a functional_aggregated_data payload."""
    bundle = _normalize_bundle(payload)
    unknown = frozenset(bundle) - _FUNCTIONAL_AGGREGATION_PANEL_KEYS
    if unknown:
        raise ValueError(
            "Unknown functional_aggregated_data panel key(s): "
            f"{', '.join(sorted(unknown))}. "
            "Keys must match aitb-dashboard case bundle panels "
            f"({', '.join(sorted(_FUNCTIONAL_AGGREGATION_PANEL_KEYS))})."
        )
    qs = orm.PatientCase.objects.all()
    for panel_key, rows in bundle.items():
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_q = _panel_row_q(panel_key, row)
            if row_q is not None:
                qs = qs.filter(row_q)
    return qs


def interpolated_sql_for_functional_aggregated_patient_case_filter(
    qs: models.QuerySet,
) -> str:
    """Human-readable SQL for the queryset filter (same shape as DEBUG logs)."""
    compiler = qs.query.get_compiler(using=qs.db)
    try:
        sql, params = compiler.as_sql()
        return _sql_with_params_interpolated_for_log(qs.db, sql, params)
    except Exception as exc:  # pragma: no cover - defensive
        return (
            f"-- as_sql failed: {exc!r}\n"
            f"-- fallback queryset representation:\n{str(qs.query)}"
        )


def count_patient_cases_matching_functional_aggregated_data(payload: Any) -> int:
    qs = patient_cases_queryset_for_functional_aggregated_data(payload)
    if logger.isEnabledFor(logging.DEBUG):
        filled = interpolated_sql_for_functional_aggregated_patient_case_filter(qs)
        logger.debug(
            "similarity count patient count SQL (compiled filter; ORM wraps in COUNT for .count()):\n%s",
            filled,
        )
    n = qs.count()
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("similarity count patient count result: %s", n)
    return n


def parse_functional_aggregated_data_query_param(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e
