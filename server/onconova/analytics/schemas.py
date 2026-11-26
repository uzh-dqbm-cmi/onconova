"""
This module defines Pydantic schemas for dashboard analytics and patient case matching
within the Onconova data platform. The schemas are used for API serialization and
validation of platform-wide statistics, monthly counts, entity-level statistics, data
completion metrics, and partial patient case bundles for matching count queries.
"""
import re
from datetime import date
from types import UnionType
from typing import Annotated, Any, ClassVar, Dict, List, Union, get_args, get_origin

from django.core.exceptions import FieldDoesNotExist
from django.db.models import Model as DjangoModel
from ninja import Field, Schema
from pydantic import BaseModel, ConfigDict, create_model
from queryable_properties.properties import QueryableProperty

from onconova.core.measures.schemas import Measure
from onconova.core.schemas import CodedConcept, Range
from onconova.core.types import Nullable
from onconova.core.utils import camel_to_snake
from onconova.oncology import models as oncology_orm


class DataPlatformStatistics(Schema):
    """
    Schema representing key statistics of the data platform.

    Attributes:
        cases (int): Total number of unique patient cases in the data platform.
        primarySites (int): Number of distinct primary anatomical sites represented.
        projects (int): Total number of research projects.
        cohorts (int): Number of defined cohorts in the platform.
        entries (int): Total number of individual data entries recorded.
        mutations (int): Total number of genetic mutations documented across all cases.
        clinicalCenters (int): Number of clinical centers contributing data.
        contributors (int): Total number of individual data contributors.
    """
    cases: int = Field(
        ...,
        title="Patient Cases",
        description="Total number of unique patient cases in the data platform.",
    )
    primarySites: int = Field(
        ...,
        title="Primary Sites",
        description="Number of distinct primary anatomical sites represented.",
    )
    projects: int = Field(
        ..., title="Projects", description="Total number of research projects."
    )
    cohorts: int = Field(
        ..., title="Cohorts", description="Number of defined cohorts in the platform."
    )
    entries: int = Field(
        ...,
        title="Data Entries",
        description="Total number of individual data entries recorded.",
    )
    mutations: int = Field(
        ...,
        title="Mutations",
        description="Total number of genetic mutations documented across all cases.",
    )
    clinicalCenters: int = Field(
        ...,
        title="Clinical Centers",
        description="Number of clinical centers contributing data.",
    )
    contributors: int = Field(
        ...,
        title="Contributors",
        description="Total number of individual data contributors.",
    )


class CountsPerMonth(Schema):
    """
    Schema representing the cumulative count of entries per month.

    Attributes:
        month (date): The month (as a date object) representing the period of data aggregation.
        cumulativeCount (int): Total number of entries accumulated up to and including the given month.
    """
    month: date = Field(
        ...,
        title="Month",
        description="The month (as date) representing the period of data aggregation.",
    )
    cumulativeCount: int = Field(
        ...,
        title="Cumulative Case Count",
        description="Total number of entries accumulated up to and including the given month.",
    )


class EntityStatistics(Schema):
    """
    Schema representing statistical data for a medical entity.

    Attributes:
        population (Optional[int]): Number of cases in the population.
        dataCompletionMedian (Optional[float]): Median percentage of case completion.
        topographyCode (Optional[str]): ICD-O-3 topography code of the entity.
        topographyGroup (Optional[str]): ICD-O-3 topography code of the entity group.
    """
    population: Nullable[int] = Field(
        default=None, title="Population", description="Number of cases"
    )
    dataCompletionMedian: Nullable[float] = Field(
        default=None,
        title="Data Completion Median",
        description="Median percentage of case completion",
    )
    topographyCode: Nullable[str] = Field(
        default=None,
        title="Topography Code",
        description="ICD-O-3 topography code of the entity",
    )
    topographyGroup: Nullable[str] = Field(
        default=None,
        title="Topography Group",
        description="ICD-O-3 topography code of the entity group",
    )


class IncompleteCategory(Schema):
    """
    Schema representing a category of data incompleteness.

    Attributes:
        category (str): The data category with cases where it is incomplete.
        cases (int): Number of cases affected with the incomplete data category.
        affectedSites (List[CodedConcept]): List of anatomical sites affected by this data incompleteness.
    """
    category: str = Field(
        ...,
        title="Incomplete Category",
        description="The data category with cases where it is incomplete.",
    )
    cases: int = Field(
        ...,
        title="Affected Cases",
        description="Number of cases affected with the incomplete data category.",
    )
    affectedSites: List[CodedConcept] = Field(
        ...,
        title="Affected Sites",
        description="List of anatomical sites affected by this data incompleteness.",
    )


class DataCompletionStatistics(Schema):
    """
    Schema representing statistics on data completion for patient cases.

    Attributes:
        totalCases (int): Total number of patient cases analyzed for data completeness.
        overallCompletion (float): Overall percentage of data categories completed across all cases.
        mostIncompleteCategories (List[IncompleteCategory]): List of the most common categories with missing data.
        completionOverTime (List[CountsPerMonth]): Historical trend of cumulative data completeness by month.
    """
    totalCases: int = Field(
        ...,
        title="Total Cases",
        description="Total number of patient cases analyzed for data completeness.",
    )
    overallCompletion: float = Field(
        ...,
        title="Overall Completion (%)",
        description="Overall percentage of data categories completed across all cases.",
    )
    mostIncompleteCategories: List[IncompleteCategory] = Field(
        ...,
        title="Most Incomplete Categories",
        description="List of the most common categories with missing data.",
    )
    completionOverTime: List[CountsPerMonth] = Field(
        ...,
        title="Completion Over Time",
        description="Historical trend of cumulative data completeness by month.",
    )


_EXCLUDED_MATCH_FIELDS: frozenset[str] = frozenset(
    {
        # Identity & audit metadata
        "id",
        "description",
        "externalSource",
        "externalSourceId",
        "createdAt",
        "updatedAt",
        "createdBy",
        "updatedBy",
        "anonymized",
        # Patient PII & personal dates
        "pseudoidentifier",
        "clinicalCenter",
        "clinicalIdentifier",
        "dateOfBirth",
        "dateOfDeath",
        "endOfRecords",
        # Case export / audit metadata
        "contributors",
        "contributorsDetails",
        "history",
        "completedDataCategories",
        "dataCompletionRate",
        # Event dates & periods (anonymized/shifted per case)
        "date",
        "assertionDate",
        "assessmentDate",
        "dateResolved",
        "progressionDate",
        "period",
    }
)

POLYMORPHIC_PARENTS: tuple[tuple[type[DjangoModel], str], ...] = (
    (oncology_orm.Staging, "stagingDomain"),
    (oncology_orm.GenomicSignature, "category"),
    (oncology_orm.TumorBoard, "category"),
)
"""Polymorphic multi-table parent models paired with their schema discriminator field name."""


def get_polymorphic_parent(
    model: type[DjangoModel] | None,
) -> tuple[type[DjangoModel], str] | None:
    """
    Return the polymorphic `(parent_model, discriminator_field)` pair for `model`.

    Args:
        model (type[DjangoModel] | None): Django ORM model class to inspect.

    Returns:
        tuple[type[DjangoModel], str] | None: Matching polymorphic parent and
            schema discriminator field name, or `None` for non-polymorphic models.
    """
    if model is None:
        return None
    for parent, discriminator in POLYMORPHIC_PARENTS:
        if issubclass(model, parent):
            return parent, discriminator
    return None


def resolve_orm_attribute(
    schema_cls: type[BaseModel],
    field_name: str,
    model: type[DjangoModel],
) -> tuple[str, Any | None, QueryableProperty | None] | None:
    """
    Resolve a camelCase schema field name to its Django ORM field or `QueryableProperty`.

    Checks the source schema's explicit alias, standard `camel_to_snake` conversion,
    and an acronym-preserving snake_case fallback (e.g. `isVUS` -> `is_vus`).

    Args:
        schema_cls (type[BaseModel]): Source or generated partial Pydantic schema class.
        field_name (str): Schema field name to resolve.
        model (type[DjangoModel]): Target Django ORM model class.

    Returns:
        tuple[str, Any | None, QueryableProperty | None] | None:
            `(orm_name, django_field, queryable_property)` where at most one of
            `django_field` or `queryable_property` is populated, or `None` if unmapped.
    """
    source_schema = getattr(schema_cls, "__source_schema__", None) or schema_cls
    field_info = source_schema.model_fields.get(field_name)
    candidates = [
        *([field_info.alias] if field_info and field_info.alias else ()),
        camel_to_snake(field_name),
        re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", field_name).lower(),
    ]
    for candidate in dict.fromkeys(candidates):
        try:
            return candidate, model._meta.get_field(candidate), None
        except FieldDoesNotExist:
            descriptor = getattr(model, candidate, None)
            prop = getattr(descriptor, "prop", descriptor)
            if isinstance(prop, QueryableProperty):
                return candidate, None, prop
    return None


class _PartialMatchingSchema(Schema):
    """Base schema for generated partial matching criteria models."""

    model_config = ConfigDict(extra="forbid")

    __orm_model__: ClassVar[type[DjangoModel] | None] = None
    __source_schema__: ClassVar[type[BaseModel] | None] = None


class PartialCodedConcept(_PartialMatchingSchema):
    """
    Partial representation of a coded concept for matching queries.

    Only `code` is required; `system` and descriptive fields are optional so
    callers may match by code alone or pass a full exported `CodedConcept`.

    Attributes:
        code (str): Unique code within a coding system that identifies a concept.
        system (Nullable[str]): Optional canonical URL of the code system.
        display (Nullable[str]): Optional human-readable description of the concept.
        version (Nullable[str]): Optional release version of the code system.
        synonyms (Nullable[List[str]]): Optional list of synonyms.
        properties (Nullable[Dict[str, Any]]): Optional additional concept properties.
    """

    code: str = Field(
        ...,
        title="Code",
        description="Unique code within a coding system that identifies a concept.",
    )
    system: Nullable[str] = Field(
        default=None,
        title="System",
        description="Canonical URL of the code system defining the concept.",
    )
    display: Nullable[str] = Field(
        default=None,
        title="Display",
        description="Human-readable description of the concept.",
    )
    version: Nullable[str] = Field(
        default=None,
        title="Version",
        description="Release version of the code system, if applicable.",
    )
    synonyms: Nullable[List[str]] = Field(
        default=None,
        title="Synonyms",
        description="List of synonyms or alternative representations of the concept.",
    )
    properties: Nullable[Dict[str, Any]] = Field(
        default=None,
        title="Properties",
        description="Additional properties associated with the concept.",
    )


class PartialRange(_PartialMatchingSchema):
    """
    Partial representation of a numeric interval for matching queries.

    Attributes:
        start (Nullable[int | float]): Lower bound of the range.
        end (Nullable[int | float]): Upper bound of the range.
    """

    start: Nullable[int | float] = Field(
        default=None,
        title="Start",
        description="The lower bound of the range.",
    )
    end: Nullable[int | float] = Field(
        default=None,
        title="End",
        description="The upper bound of the range.",
    )


_SPECIAL_PARTIAL_TYPES: tuple[tuple[type[BaseModel], type[BaseModel]], ...] = (
    (CodedConcept, PartialCodedConcept),
    (Range, PartialRange),
    (Measure, Measure),
)


def _transform_partial_annotation(
    annotation: Any,
    excluded_fields: frozenset[str],
    cache: dict[type[BaseModel], type[Schema]],
) -> Any:
    """Recursively transform a Pydantic field annotation into its partial-schema equivalent."""
    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Annotated:
        return _transform_partial_annotation(args[0], excluded_fields, cache)

    if origin in (Union, UnionType):
        members = [
            _transform_partial_annotation(arg, excluded_fields, cache)
            for arg in args
            if arg is not type(None)
        ]
        return members[0] if len(members) == 1 else Union[tuple(members)]  # type: ignore[valid-type]

    if origin is list:
        return (
            List[_transform_partial_annotation(args[0], excluded_fields, cache)]  # type: ignore[valid-type]
            if args
            else List[Any]
        )

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        for source_type, target_type in _SPECIAL_PARTIAL_TYPES:
            if issubclass(annotation, source_type):
                return target_type
        return make_partial_schema(annotation, excluded_fields=excluded_fields, cache=cache)

    return annotation


def make_partial_schema(
    schema_cls: type[BaseModel],
    *,
    excluded_fields: frozenset[str] = _EXCLUDED_MATCH_FIELDS,
    cache: dict[type[BaseModel], type[Schema]] | None = None,
) -> type[_PartialMatchingSchema]:
    """
    Dynamically creates a partial matching schema from an existing Pydantic schema.

    Every non-excluded, ORM-backed field (and every nested sub-schema) becomes
    optional while retaining field titles, descriptions, OpenAPI metadata
    extensions, and a link to the source schema's ORM model (`__orm_model__`).

    Args:
        schema_cls (type[BaseModel]): Source Pydantic schema class to mirror.
        excluded_fields (frozenset[str]): Field names omitted from the partial schema.
        cache (dict[type[BaseModel], type[Schema]] | None): Optional memoization cache
            shared across recursive calls to avoid duplicate model creation.

    Returns:
        type[_PartialMatchingSchema]: Generated partial Pydantic schema class.
    """
    cache = {} if cache is None else cache
    if schema_cls in cache:
        return cache[schema_cls]  # type: ignore[return-value]

    orm_model = getattr(schema_cls, "__orm_model__", None)
    poly = get_polymorphic_parent(orm_model)
    discriminator_field = poly[1] if poly is not None else None
    field_definitions: dict[str, tuple[Any, Any]] = {}
    for field_name, field_info in schema_cls.model_fields.items():
        if field_name in excluded_fields or field_name.endswith(("Id", "Ids")):
            continue
        if (
            orm_model is not None
            and field_name != discriminator_field
            and resolve_orm_attribute(schema_cls, field_name, orm_model) is None
        ):
            continue

        transformed = _transform_partial_annotation(
            field_info.annotation, excluded_fields=excluded_fields, cache=cache
        )
        title = field_info.title or camel_to_snake(field_name).replace("_", " ").capitalize()
        field_kwargs: dict[str, Any] = {
            "default": None,
            "title": title,
            "description": field_info.description or f"Matching criteria for {title.lower()}.",
        }
        if isinstance(field_info.json_schema_extra, dict):
            field_kwargs["json_schema_extra"] = dict(field_info.json_schema_extra)
        if pattern := next(
            (p for meta in field_info.metadata if (p := getattr(meta, "pattern", None))),
            None,
        ):
            field_kwargs["pattern"] = pattern

        field_definitions[field_name] = (
            Nullable[transformed],  # type: ignore[valid-type]
            Field(**field_kwargs),
        )

    partial_cls: type[_PartialMatchingSchema] = create_model(
        f"Partial{schema_cls.__name__}",
        __base__=_PartialMatchingSchema,
        __module__=__name__,
        **field_definitions,
    )
    partial_cls.__doc__ = f"Partial matching criteria schema derived from `{schema_cls.__name__}`."
    partial_cls.__orm_model__ = orm_model
    partial_cls.__source_schema__ = schema_cls
    cache[schema_cls] = partial_cls
    return partial_cls


def _build_partial_patient_case_bundle() -> type[_PartialMatchingSchema]:
    from onconova.interoperability.schemas import PatientCaseBundle

    return make_partial_schema(PatientCaseBundle)


PartialPatientCaseBundle = _build_partial_patient_case_bundle()
"""Partial matching criteria schema derived from `PatientCaseBundle`."""


class MatchingCountRequest(Schema):
    """
    Request payload for counting patient cases matching a partial patient case bundle.

    Attributes:
        bundle (PartialPatientCaseBundle): Partial patient case bundle defining the matching criteria.
    """

    model_config = ConfigDict(extra="forbid")

    bundle: PartialPatientCaseBundle = Field(  # type: ignore[valid-type]
        ...,
        title="Patient case bundle",
        description="Partial patient case bundle specifying the criteria that matching patient cases must satisfy.",
    )
