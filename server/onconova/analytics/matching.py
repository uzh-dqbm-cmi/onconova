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
    get_orm_field_or_qprop,
    get_polymorphic_parent,
    resolve_orm_field_name,
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
# static score-to-LOINC mappings (avoids nested OuterRef ambiguity inside EXISTS).
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


def _is_coded_concept_target(django_field: Any | None, qprop: QueryableProperty | None) -> bool:
    """Return True if a field or SubqueryObjectProperty points to a `CodedConcept` model."""
    if django_field is not None and getattr(django_field, "is_relation", False):
        related = getattr(django_field, "related_model", None)
        return isinstance(related, type) and issubclass(related, CodedConcept)
    if isinstance(qprop, SubqueryObjectProperty):
        target = getattr(qprop, "_subquery_model", None)
        return isinstance(target, type) and issubclass(target, CodedConcept)
    return False


def _get_measurement_field(
    django_field: Any | None, qprop: QueryableProperty | None
) -> MeasurementField | None:
    """Return the `MeasurementField` backing a model field or `AnnotationProperty`, if any."""
    if isinstance(django_field, MeasurementField):
        return django_field
    if isinstance(qprop, AnnotationProperty):
        output_field = getattr(getattr(qprop, "annotation", None), "output_field", None)
        if isinstance(output_field, MeasurementField):
            return output_field
    return None


def _is_measurement_target(django_field: Any | None, qprop: QueryableProperty | None) -> bool:
    """Return True if a field or AnnotationProperty represents a `MeasurementField`."""
    return _get_measurement_field(django_field, qprop) is not None


def _model_supports_field_value(
    model: type[models.Model],
    schema_cls: type[BaseModel],
    field_name: str,
    value: Any,
) -> bool:
    """Check whether `model` has a compatible field or property for `(field_name, value)`."""
    orm_name = resolve_orm_field_name(schema_cls, field_name, model)
    if orm_name is None:
        return False
    django_field, qprop = get_orm_field_or_qprop(model, orm_name)
    is_coded = _is_coded_concept_target(django_field, qprop)
    if isinstance(value, PartialCodedConcept):
        return is_coded
    if isinstance(value, list) and value and isinstance(value[0], PartialCodedConcept):
        return is_coded
    if is_coded:
        return False
    if isinstance(value, Measure):
        return _is_measurement_target(django_field, qprop)
    if (
        isinstance(django_field, models.IntegerField)
        and isinstance(value, float)
        and not value.is_integer()
    ):
        return False
    return True


def _coded_concept_q(orm_name: str, concept: PartialCodedConcept) -> Q:
    """Build a `Q` filter matching a `CodedConcept` by code (and system if provided)."""
    q = Q(**{f"{orm_name}__code": concept.code})
    if concept.system:
        q &= Q(**{f"{orm_name}__system": concept.system})
    return q


def _age_bin_q(orm_name: str, age_bin: AgeBin) -> Q:
    """Translate an `AgeBin` label (`<20`, `20-24`, `90+`) into an ORM range `Q`."""
    label = age_bin.value
    if label.startswith("<"):
        return Q(**{f"{orm_name}__lt": int(label[1:])})
    if label.endswith("+"):
        return Q(**{f"{orm_name}__gte": int(label[:-1])})
    low, high = (int(bound) for bound in label.split("-", 1))
    return Q(**{f"{orm_name}__range": (low, high)})


def _build_scalar_q(
    model: type[models.Model],
    orm_name: str,
    value: Any,
    django_field: Any | None,
    qprop: QueryableProperty | None,
) -> Q:
    """Build a Django `Q` expression for a single scalar, coded, range, or measure value."""
    if isinstance(value, PartialCodedConcept):
        score_mapping = _SCORE_INTERPRETATION_MAPS.get((model, orm_name))
        if score_mapping is not None:
            score_field, code_to_score = score_mapping
            score = code_to_score.get(value.code)
            if score is None:
                return Q(pk__in=[])
            return Q(**{score_field: score})
        return _coded_concept_q(orm_name, value)

    if isinstance(value, AgeBin):
        return _age_bin_q(orm_name, value)

    if isinstance(value, Measure):
        measurement_field = _get_measurement_field(django_field, qprop)
        if measurement_field is None:
            raise ValueError(f"Field '{orm_name}' does not support measurement values.")
        try:
            measurement = measurement_field.measurement(**{value.unit: value.value})
        except Exception as exc:
            raise ValueError(
                f"Invalid unit '{value.unit}' for measurement field '{orm_name}'."
            ) from exc
        return Q(**{orm_name: float(measurement.standard)})

    if isinstance(value, PartialRange):
        return Q(**{orm_name: PostgresRange(value.start, value.end, bounds="[)")})

    if isinstance(value, list):
        return Q(**{f"{orm_name}__contains": value})

    if isinstance(value, str):
        if django_field is not None:
            if getattr(django_field, "choices", None):
                return Q(**{orm_name: value})
            return Q(**{f"{orm_name}__iexact": value})
        if isinstance(qprop, AnnotationProperty) and isinstance(
            getattr(qprop, "annotation", None), StringAgg
        ):
            return Q(**{f"{orm_name}__icontains": value})
        return Q(**{f"{orm_name}__iexact": value})

    return Q(**{orm_name: value})


def _entry_discriminator_field(
    entry: BaseModel,
    default_model: type[models.Model] | None = None,
) -> str | None:
    """Return the polymorphic discriminator field name for `entry`, if any."""
    model = getattr(entry.__class__, "__orm_model__", None) or default_model
    poly = get_polymorphic_parent(model)
    return poly[1] if poly is not None else None


def _active_entry_items(
    entry: BaseModel,
    *,
    discriminator: str | None = None,
) -> list[tuple[str, Any]]:
    """Return non-discriminator fields explicitly set on `entry` with non-empty values."""
    items: list[tuple[str, Any]] = []
    for field_name in entry.model_fields_set:
        if field_name == discriminator:
            continue
        value = getattr(entry, field_name)
        if value is None or value == []:
            continue
        items.append((field_name, value))
    return items


def _has_explicit_discriminator(
    entry: BaseModel,
    discriminator: str | None,
) -> bool:
    """Return True if `entry` explicitly specifies a non-null polymorphic discriminator."""
    return bool(
        discriminator
        and discriminator in entry.model_fields_set
        and getattr(entry, discriminator, None) is not None
    )


def _apply_entry_filters(
    qs: QuerySet,
    model: type[models.Model],
    entry: BaseModel,
) -> tuple[QuerySet, bool]:
    """
    Apply all explicitly-set non-empty criteria from `entry` to `qs`.

    Handles scalar fields, coded concept foreign keys and M2M relations,
    queryable properties, and recursive reverse one-to-many sub-resources.
    Raises `ValueError` if any set criterion cannot be mapped to `model`
    so invalid criteria are never silently dropped.
    """
    matched = False
    schema_cls = entry.__class__
    poly = get_polymorphic_parent(model)
    discriminator = poly[1] if poly is not None else None

    for field_name, value in _active_entry_items(entry, discriminator=discriminator):
        orm_name = resolve_orm_field_name(schema_cls, field_name, model)
        if orm_name is None:
            raise ValueError(
                f"Field '{field_name}' is not a queryable attribute on {model.__name__}."
            )
        django_field, qprop = get_orm_field_or_qprop(model, orm_name)

        if isinstance(value, list):
            # Many-to-many CodedConcept relation: AND across every listed concept.
            if (
                django_field is not None
                and getattr(django_field, "many_to_many", False)
                and _is_coded_concept_target(django_field, None)
            ):
                for item in value:
                    qs = qs.filter(_coded_concept_q(orm_name, item))
                    matched = True
                continue

            # Reverse one-to-many child resources: independent nested EXISTS per entry.
            if (
                django_field is not None
                and getattr(django_field, "one_to_many", False)
                and django_field.related_model is not None
            ):
                child_model = django_field.related_model
                parent_fk = f"{django_field.field.name}_id"
                for child_entry in value:
                    if not isinstance(child_entry, BaseModel):
                        continue
                    child_q = _child_entry_to_exists_q(
                        child_entry,
                        parent_fk=parent_fk,
                        default_model=child_model,
                    )
                    if child_q is not None:
                        qs = qs.filter(child_q)
                        matched = True
                continue

            # ArrayField / array-valued AnnotationProperty (e.g. chromosomes, regions).
            qs = qs.filter(_build_scalar_q(model, orm_name, value, django_field, qprop))
            matched = True
            continue

        qs = qs.filter(_build_scalar_q(model, orm_name, value, django_field, qprop))
        matched = True

    return qs, matched


def _resolve_candidate_models(
    entry: BaseModel,
    active_items: list[tuple[str, Any]],
    default_model: type[models.Model] | None = None,
) -> list[type[models.Model]]:
    """
    Determine the ORM model(s) to query for `entry`.

    For polymorphic multi-table hierarchies (`Staging`, `GenomicSignature`,
    `TumorBoard`), an explicit discriminator narrows matching to that concrete
    subclass; omitting the discriminator matches across all compatible subclasses
    (or the shared parent model when all provided fields live on the parent).
    Raises `ValueError` when non-empty criteria cannot be satisfied by any model.
    """
    model: type[models.Model] | None = (
        getattr(entry.__class__, "__orm_model__", None) or default_model
    )
    if model is None:
        if active_items:
            fields = ", ".join(sorted(fname for fname, _ in active_items))
            raise ValueError(
                f"Criteria '{fields}' on {entry.__class__.__name__} do not map to an ORM model."
            )
        return []

    poly = get_polymorphic_parent(model)
    if poly is not None:
        parent, discriminator = poly
        if model is not parent:
            if _has_explicit_discriminator(entry, discriminator):
                return [model]

            if not active_items:
                return []

            # Prefer the shared parent table when every criterion lives directly on it.
            if all(
                _model_supports_field_value(parent, entry.__class__, fname, val)
                for fname, val in active_items
            ):
                return [parent]

            candidates = [
                subcls
                for subcls in parent.__subclasses__()
                if all(
                    _model_supports_field_value(subcls, entry.__class__, fname, val)
                    for fname, val in active_items
                )
            ]
            if not candidates:
                fields = ", ".join(sorted(fname for fname, _ in active_items))
                raise ValueError(
                    f"No {parent.__name__} subtype supports the combined criteria: {fields}."
                )
            return candidates

    return [model]


def _child_entry_to_exists_q(
    entry: BaseModel,
    *,
    parent_fk: str = "case_id",
    default_model: type[models.Model] | None = None,
) -> Q | None:
    """Build an `EXISTS` (or OR of `EXISTS` for polymorphic entries) `Q` for `entry`."""
    discriminator = _entry_discriminator_field(entry, default_model=default_model)
    active_items = _active_entry_items(entry, discriminator=discriminator)
    has_discriminator = _has_explicit_discriminator(entry, discriminator)
    if not active_items and not has_discriminator:
        return None

    candidates = _resolve_candidate_models(
        entry, active_items, default_model=default_model
    )
    exists_parts: list[Q] = []
    for candidate_model in candidates:
        subqs = candidate_model.objects.filter(**{parent_fk: OuterRef("pk")})
        subqs, matched = _apply_entry_filters(subqs, candidate_model, entry)
        if matched or has_discriminator:
            exists_parts.append(Q(Exists(subqs)))

    if not exists_parts:
        return None
    return functools.reduce(operator.or_, exists_parts)


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
    qs: QuerySet[orm.PatientCase] = orm.PatientCase.objects.all()
    qs, _ = _apply_entry_filters(qs, orm.PatientCase, bundle)
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
