"""
Django ORM matching engine for counting patient cases that satisfy a partial
patient case bundle.

Each non-empty resource entry in the bundle adds an independent `EXISTS`
requirement against `PatientCase` (logical AND across entries and resource
categories). Nested sub-resources (such as systemic therapy medications or
radiotherapy dosages/settings) are matched recursively via nested `EXISTS`
subqueries so multi-field criteria constrain the same child record without
row fanout.
"""

from __future__ import annotations

import functools
import logging
import operator
from typing import Any

from django.contrib.postgres.aggregates import StringAgg
from django.core.exceptions import EmptyResultSet
from django.db import models
from django.db.models import Exists, OuterRef, Q, QuerySet
from psycopg.types.range import Range as PostgresRange
from pydantic import BaseModel
from queryable_properties.properties import (
    AnnotationProperty,
    QueryableProperty,
    SubqueryObjectProperty,
)

from onconova.analytics.schemas import (
    PartialCodedConcept,
    PartialPatientCaseBundle,
    PartialRange,
    get_polymorphic_parent,
    resolve_orm_attribute,
)
from onconova.core.measures.fields import MeasurementField
from onconova.core.measures.schemas import Measure
from onconova.core.types import AgeBin
from onconova.oncology import models as orm
from onconova.oncology.models.performance_status import (
    ECOG_INTEPRETATION,
    KARNOFSKY_INTEPRETATION,
)
from onconova.terminology.models import CodedConcept

logger = logging.getLogger(__name__)

# Direct score lookup tables for SubqueryObjectProperty interpretations backed by
# static score-to-LOINC mappings (avoids correlated terminology subqueries inside EXISTS).
_SCORE_INTERPRETATION_MAPS: dict[
    tuple[type[models.Model], str], tuple[str, dict[str, int]]
] = {
    (orm.PerformanceStatus, "ecog_interpretation"): (
        "ecog_score",
        {code: score for score, code in ECOG_INTEPRETATION.items()},
    ),
    (orm.PerformanceStatus, "karnofsky_interpretation"): (
        "karnofsky_score",
        {code: score for score, code in KARNOFSKY_INTEPRETATION.items()},
    ),
}


def _is_coded_target(django_field: Any | None, qprop: QueryableProperty | None) -> bool:
    """Return True if a field or `SubqueryObjectProperty` targets a `CodedConcept` model."""
    target = (
        getattr(django_field, "related_model", None)
        if getattr(django_field, "is_relation", False)
        else getattr(qprop, "_subquery_model", None)
        if isinstance(qprop, SubqueryObjectProperty)
        else None
    )
    return isinstance(target, type) and issubclass(target, CodedConcept)


def _measurement_field(
    django_field: Any | None, qprop: QueryableProperty | None
) -> MeasurementField | None:
    """Return the `MeasurementField` backing a model field or `AnnotationProperty`, if any."""
    candidate = (
        django_field
        if isinstance(django_field, MeasurementField)
        else getattr(getattr(qprop, "annotation", None), "output_field", None)
        if isinstance(qprop, AnnotationProperty)
        else None
    )
    return candidate if isinstance(candidate, MeasurementField) else None


def _coded_concept_q(orm_name: str, concept: PartialCodedConcept) -> Q:
    """Build a `Q` filter matching a `CodedConcept` by code (and system if provided)."""
    filters = {f"{orm_name}__code": concept.code}
    if concept.system:
        filters[f"{orm_name}__system"] = concept.system
    return Q(**filters)


def _model_supports_field_value(
    model: type[models.Model],
    schema_cls: type[BaseModel],
    field_name: str,
    value: Any,
) -> bool:
    """Check whether `model` has a compatible ORM attribute for `(field_name, value)`."""
    resolved = resolve_orm_attribute(schema_cls, field_name, model)
    if resolved is None:
        return False
    _, django_field, qprop = resolved
    is_coded = _is_coded_target(django_field, qprop)
    if isinstance(value, PartialCodedConcept) or (
        isinstance(value, list) and value and isinstance(value[0], PartialCodedConcept)
    ):
        return is_coded
    if is_coded:
        return False
    if isinstance(value, Measure):
        return _measurement_field(django_field, qprop) is not None
    return not (
        isinstance(django_field, models.IntegerField)
        and isinstance(value, float)
        and not value.is_integer()
    )


def _build_value_q(
    model: type[models.Model],
    orm_name: str,
    value: Any,
    django_field: Any | None,
    qprop: QueryableProperty | None,
) -> Q:
    """Build a Django `Q` expression for a single scalar, coded, range, array, or measure criterion."""
    match value:
        case PartialCodedConcept():
            if mapping := _SCORE_INTERPRETATION_MAPS.get((model, orm_name)):
                score_field, code_to_score = mapping
                score = code_to_score.get(value.code)
                return Q(pk__in=[]) if score is None else Q(**{score_field: score})
            return _coded_concept_q(orm_name, value)

        case AgeBin():
            label = value.value
            if label.startswith("<"):
                return Q(**{f"{orm_name}__lt": int(label[1:])})
            if label.endswith("+"):
                return Q(**{f"{orm_name}__gte": int(label[:-1])})
            low, high = map(int, label.split("-", 1))
            return Q(**{f"{orm_name}__range": (low, high)})

        case Measure():
            mfield = _measurement_field(django_field, qprop)
            if mfield is None:
                raise ValueError(f"Field '{orm_name}' does not support measurement values.")
            try:
                standard_value = float(mfield.measurement(**{value.unit: value.value}).standard)
            except Exception as exc:
                raise ValueError(
                    f"Invalid unit '{value.unit}' for measurement field '{orm_name}'."
                ) from exc
            return Q(**{orm_name: standard_value})

        case PartialRange():
            return Q(**{orm_name: PostgresRange(value.start, value.end, bounds="[)")})

        case list():
            return Q(**{f"{orm_name}__contains": value})

        case str():
            if getattr(django_field, "choices", None):
                return Q(**{orm_name: value})
            if isinstance(qprop, AnnotationProperty) and isinstance(
                getattr(qprop, "annotation", None), StringAgg
            ):
                return Q(**{f"{orm_name}__icontains": value})
            return Q(**{f"{orm_name}__iexact": value})

        case _:
            return Q(**{orm_name: value})


def _entry_criteria(
    entry: BaseModel, model: type[models.Model] | None
) -> tuple[list[tuple[str, Any]], bool]:
    """Extract non-empty, non-discriminator criteria and whether an explicit discriminator is set."""
    poly = get_polymorphic_parent(model)
    discriminator = poly[1] if poly is not None else None
    has_discriminator = bool(
        discriminator
        and discriminator in entry.model_fields_set
        and getattr(entry, discriminator, None) is not None
    )
    items = [
        (name, value)
        for name in entry.model_fields_set
        if name != discriminator
        and (value := getattr(entry, name)) is not None
        and value != []
    ]
    return items, has_discriminator


def _apply_entry_filters(
    qs: QuerySet,
    model: type[models.Model],
    entry: BaseModel,
    *,
    active_items: list[tuple[str, Any]] | None = None,
) -> tuple[QuerySet, bool]:
    """
    Apply all explicitly-set non-empty criteria from `entry` to `qs`.

    Raises `ValueError` if any set criterion cannot be mapped to `model`.
    """
    matched = False
    items = _entry_criteria(entry, model)[0] if active_items is None else active_items

    for field_name, value in items:
        resolved = resolve_orm_attribute(entry.__class__, field_name, model)
        if resolved is None:
            raise ValueError(
                f"Field '{field_name}' is not a queryable attribute on {model.__name__}."
            )
        orm_name, django_field, qprop = resolved

        if (
            isinstance(value, list)
            and getattr(django_field, "many_to_many", False)
            and _is_coded_target(django_field, None)
        ):
            for item in value:
                qs = qs.filter(_coded_concept_q(orm_name, item))
                matched = True
        elif (
            isinstance(value, list)
            and getattr(django_field, "one_to_many", False)
            and django_field.related_model is not None
        ):
            parent_fk = f"{django_field.field.name}_id"
            for child_entry in value:
                if isinstance(child_entry, BaseModel) and (
                    child_q := _child_entry_to_exists_q(
                        child_entry,
                        parent_fk=parent_fk,
                        default_model=django_field.related_model,
                    )
                ) is not None:
                    qs = qs.filter(child_q)
                    matched = True
        else:
            qs = qs.filter(_build_value_q(model, orm_name, value, django_field, qprop))
            matched = True

    return qs, matched


def _resolve_candidate_models(
    entry: BaseModel,
    model: type[models.Model] | None,
    active_items: list[tuple[str, Any]],
    has_discriminator: bool,
) -> list[type[models.Model]]:
    """Determine concrete or parent ORM model(s) to query for `entry`, failing closed on mismatch."""
    if model is None:
        fields = ", ".join(sorted(name for name, _ in active_items))
        raise ValueError(
            f"Criteria '{fields}' on {entry.__class__.__name__} do not map to an ORM model."
        )

    poly = get_polymorphic_parent(model)
    if poly is None or model is poly[0] or has_discriminator:
        return [model]

    parent = poly[0]
    supports = lambda cls: all(
        _model_supports_field_value(cls, entry.__class__, name, val)
        for name, val in active_items
    )
    if supports(parent):
        return [parent]
    if candidates := [sub for sub in parent.__subclasses__() if supports(sub)]:
        return candidates

    fields = ", ".join(sorted(name for name, _ in active_items))
    raise ValueError(
        f"No {parent.__name__} subtype supports the combined criteria: {fields}."
    )


def _child_entry_to_exists_q(
    entry: BaseModel,
    *,
    parent_fk: str = "case_id",
    default_model: type[models.Model] | None = None,
) -> Q | None:
    """Build an `EXISTS` (or OR of `EXISTS` for polymorphic entries) `Q` for `entry`."""
    model = getattr(entry.__class__, "__orm_model__", None) or default_model
    active_items, has_discriminator = _entry_criteria(entry, model)
    if not active_items and not has_discriminator:
        return None

    candidates = _resolve_candidate_models(entry, model, active_items, has_discriminator)
    exists_parts: list[Q] = []
    for candidate in candidates:
        subqs, matched = _apply_entry_filters(
            candidate.objects.filter(**{parent_fk: OuterRef("pk")}),
            candidate,
            entry,
            active_items=active_items,
        )
        if matched or has_discriminator:
            exists_parts.append(Q(Exists(subqs)))

    return functools.reduce(operator.or_, exists_parts) if exists_parts else None


def get_matching_patient_cases(
    bundle: PartialPatientCaseBundle,  # type: ignore[valid-type]
) -> QuerySet[orm.PatientCase]:
    """
    Build a Django ORM `QuerySet` of `PatientCase` rows matching `bundle`.

    Every top-level patient attribute set on `bundle` constrains `PatientCase`
    directly, and every non-empty resource entry in any bundle list adds an
    independent `EXISTS` condition (logical AND across entries and resources).

    Args:
        bundle (PartialPatientCaseBundle): Validated partial patient case bundle
            specifying the matching criteria.

    Returns:
        QuerySet[orm.PatientCase]: Filtered `PatientCase` queryset.

    Raises:
        ValueError: If any set criterion cannot be mapped to a queryable ORM
            attribute, has an invalid measurement unit, or has no compatible
            polymorphic model subtype.
    """
    qs, _ = _apply_entry_filters(orm.PatientCase.objects.all(), orm.PatientCase, bundle)
    return qs


def count_matching_patient_cases(
    bundle: PartialPatientCaseBundle,  # type: ignore[valid-type]
) -> int:
    """
    Count `PatientCase` rows matching all criteria in `bundle`.

    Executes the count purely through the Django ORM. When the module logger is
    configured at `DEBUG` level, the parameterized ORM SQL query template with
    its bound parameter count (without logging bound parameter values) and the
    resulting integer count are written to the server log.

    Args:
        bundle (PartialPatientCaseBundle): Validated partial patient case bundle
            specifying the matching criteria.

    Returns:
        int: Number of matching `PatientCase` records.

    Raises:
        ValueError: If any set criterion cannot be mapped to a queryable ORM
            attribute, has an invalid measurement unit, or has no compatible
            polymorphic model subtype.
    """
    qs = get_matching_patient_cases(bundle)
    if logger.isEnabledFor(logging.DEBUG):
        try:
            sql_template, params = qs.query.sql_with_params()
        except EmptyResultSet:
            logger.debug(
                "Matching patient cases SQL template (bound parameters: 0): <empty result set>"
            )
        else:
            logger.debug(
                "Matching patient cases SQL template (bound parameters: %d):\n%s",
                len(params),
                sql_template,
            )
    count = qs.count()
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Matching patient cases count: %s", count)
    return count
