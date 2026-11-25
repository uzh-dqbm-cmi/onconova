import logging
from typing import ClassVar
from unittest import mock

from django.db.models import QuerySet
from django.test import SimpleTestCase, TestCase
from parameterized import parameterized
from pydantic import Field, ValidationError

import onconova.core.measures as measures
import onconova.terminology.models as terminology
from onconova.analytics.matching import (
    _apply_entry_filters,
    _child_entry_to_exists_q,
    count_matching_patient_cases,
    get_matching_patient_cases,
)
from onconova.analytics.schemas import (
    MatchingCountRequest,
    PartialPatientCaseBundle,
    _PartialMatchingSchema,
    make_partial_schema,
)
from onconova.api import api
from onconova.core.schemas import BaseSchema
from onconova.core.types import Nullable
from onconova.oncology import models as orm
from onconova.oncology import schemas as oncology_schemas
from onconova.oncology.models.genomic_signature import (
    TumorMutationalBurdenStatusChoices,
)
from onconova.oncology.models.patient_case import PatientCaseVitalStatusChoices
from onconova.oncology.models.radiotherapy import RadiotherapyIntentChoices
from onconova.oncology.models.surgery import SurgeryIntentChoices
from onconova.oncology.models.systemic_therapy import SystemicTherapyIntentChoices
from onconova.oncology.models.therapy_line import TherapyLineIntentChoices
from onconova.tests.common import GET_HTTP_SCENARIOS, ApiControllerTestMixin
from onconova.tests.factories import (
    BreslowDepthFactory,
    ECOGPerformanceStatusFactory,
    GenomicVariantFactory,
    KarnofskyPerformanceStatusFactory,
    LymphomaStagingFactory,
    MicrosatelliteInstabilityFactory,
    MolecularTherapeuticRecommendationFactory,
    MolecularTumorBoardFactory,
    PatientCaseFactory,
    PrimaryNeoplasticEntityFactory,
    RadiotherapyFactory,
    RadiotherapySettingFactory,
    SurgeryFactory,
    SystemicTherapyFactory,
    SystemicTherapyMedicationFactory,
    TNMStagingFactory,
    TherapyLineFactory,
    TumorBoardFactory,
    TumorMarkerTestFactory,
    TumorMutationalBurdenFactory,
)


def _count(bundle_data: dict) -> int:
    """Validate `bundle_data` into a `PartialPatientCaseBundle` and count matches."""
    return count_matching_patient_cases(
        PartialPatientCaseBundle.model_validate(bundle_data)
    )


def _assert_debug_sql_template_logging(
    test_case: SimpleTestCase,
    bundle: PartialPatientCaseBundle,  # type: ignore[valid-type]
    *,
    expected_count: int,
    forbidden_values: tuple[str, ...],
) -> None:
    """Assert DEBUG logs emit the parameterized SQL template and param count without bound values, and INFO stays silent."""
    expected_sql, expected_params = get_matching_patient_cases(
        bundle
    ).query.sql_with_params()
    interpolated_sql = str(get_matching_patient_cases(bundle).query)
    test_case.assertGreater(len(expected_params), 0)
    test_case.assertIn("%s", expected_sql)

    with test_case.assertLogs(
        "onconova.analytics.matching", level=logging.DEBUG
    ) as cm:
        count = count_matching_patient_cases(bundle)
    test_case.assertEqual(count, expected_count)

    sql_logs = [message for message in cm.output if "SELECT" in message]
    test_case.assertEqual(
        len(sql_logs),
        1,
        f"Expected one SQL template trace in DEBUG logs, got: {cm.output}",
    )
    sql_log = sql_logs[0]
    test_case.assertIn(expected_sql, sql_log)
    test_case.assertIn("%s", sql_log)
    test_case.assertIn(f"bound parameters: {len(expected_params)}", sql_log)
    for forbidden in forbidden_values:
        test_case.assertNotIn(forbidden, sql_log)
    test_case.assertNotIn(repr(expected_params), sql_log)
    test_case.assertNotIn(interpolated_sql, sql_log)
    test_case.assertTrue(
        any(
            f"Matching patient cases count: {expected_count}" in message
            for message in cm.output
        ),
        f"Expected integer count in DEBUG logs, got: {cm.output}",
    )

    with test_case.assertNoLogs("onconova.analytics.matching", level=logging.INFO):
        count_matching_patient_cases(bundle)


class TestMatchingSchemaValidation(SimpleTestCase):
    """Tests for strict partial-bundle schema validation and PII exclusion."""

    def test_coded_concept_accepts_code_only_and_full_concept(self):
        minimal = MatchingCountRequest.model_validate(
            {"bundle": {"gender": {"code": "male"}}}
        )
        self.assertEqual(minimal.bundle.gender.code, "male")
        self.assertIsNone(minimal.bundle.gender.system)

        full = MatchingCountRequest.model_validate(
            {
                "bundle": {
                    "gender": {
                        "code": "male",
                        "system": "http://hl7.org/fhir/administrative-gender",
                        "display": "Male",
                    }
                }
            }
        )
        self.assertEqual(
            full.bundle.gender.system, "http://hl7.org/fhir/administrative-gender"
        )

        with self.assertRaises(ValidationError):
            MatchingCountRequest.model_validate({"bundle": {"gender": {}}})

    @parameterized.expand(
        [
            ("unknown_panel", {"bundle": {"notAPanel": []}}),
            ("string_bundle", {"bundle": "{}"}),
            ("missing_bundle", {}),
            ("pii_clinical_identifier", {"bundle": {"clinicalIdentifier": "PAT-1"}}),
            ("pii_pseudoidentifier", {"bundle": {"pseudoidentifier": "A.1234.567.89"}}),
            ("pii_date_of_birth", {"bundle": {"dateOfBirth": "1980-01-01"}}),
            ("pii_clinical_center", {"bundle": {"clinicalCenter": "Hospital A"}}),
            (
                "excluded_case_id",
                {
                    "bundle": {
                        "neoplasticEntities": [
                            {"caseId": "00000000-0000-0000-0000-000000000000"}
                        ]
                    }
                },
            ),
            (
                "excluded_therapy_line_id",
                {
                    "bundle": {
                        "radiotherapies": [
                            {"therapyLineId": "00000000-0000-0000-0000-000000000000"}
                        ]
                    }
                },
            ),
            (
                "excluded_event_date",
                {"bundle": {"stagings": [{"date": "2024-01-01"}]}},
            ),
            (
                "non_coded_enum_wrapped_as_dict",
                {
                    "bundle": {
                        "neoplasticEntities": [{"relationship": {"code": "primary"}}]
                    }
                },
            ),
            (
                "mixed_staging_subclass_fields",
                {
                    "bundle": {
                        "stagings": [
                            {"primaryTumor": {"code": "T2"}, "bulky": True}
                        ]
                    }
                },
            ),
            (
                "mismatched_staging_discriminator_and_field",
                {
                    "bundle": {
                        "stagings": [{"stagingDomain": "tnm", "bulky": True}]
                    }
                },
            ),
            (
                "non_orm_schema_field_hgvs_version",
                {"bundle": {"genomicVariants": [{"hgvsVersion": 2}]}},
            ),
        ]
    )
    def test_rejects_invalid_or_identifying_payloads(self, _name, payload):
        with self.assertRaises(ValidationError):
            MatchingCountRequest.model_validate(payload)

    def test_make_partial_schema_omits_unmapped_non_orm_fields(self):
        partial_gv = make_partial_schema(oncology_schemas.GenomicVariant)
        self.assertNotIn("hgvsVersion", partial_gv.model_fields)

        class SyntheticSchema(BaseSchema):
            __orm_model__ = orm.NeoplasticEntity

            relationship: Nullable[str] = Field(
                default=None, title="Relationship", description="Entity relationship."
            )
            computedOnly: Nullable[str] = Field(
                default=None, title="Computed", description="Field without an ORM column."
            )

        partial_synthetic = make_partial_schema(SyntheticSchema)
        self.assertIn("relationship", partial_synthetic.model_fields)
        self.assertNotIn("computedOnly", partial_synthetic.model_fields)

    def test_unmapped_field_raises_value_error_instead_of_skipping(self):
        class UnmappedEntry(_PartialMatchingSchema):
            __orm_model__: ClassVar = orm.NeoplasticEntity
            ghostField: Nullable[str] = None

        entry = UnmappedEntry(ghostField="unexpected")
        with self.assertRaises(ValueError):
            _apply_entry_filters(
                orm.NeoplasticEntity.objects.all(), orm.NeoplasticEntity, entry
            )

    def test_unsatisfiable_polymorphic_entry_raises_value_error(self):
        class MixedStagingEntry(_PartialMatchingSchema):
            __orm_model__: ClassVar = orm.TNMStaging
            primaryTumor: Nullable[str] = None
            bulky: Nullable[bool] = None

        entry = MixedStagingEntry(primaryTumor="T2", bulky=True)
        with self.assertRaises(ValueError):
            _child_entry_to_exists_q(entry)

    def test_unmapped_model_with_active_criteria_raises_value_error(self):
        class OrphanEntry(_PartialMatchingSchema):
            __orm_model__: ClassVar = None
            label: Nullable[str] = None

        with self.assertRaises(ValueError):
            _child_entry_to_exists_q(OrphanEntry(label="x"))

    def test_invalid_measurement_unit_raises_value_error(self):
        bundle = PartialPatientCaseBundle.model_validate(
            {
                "stagings": [
                    {
                        "stagingDomain": "breslow",
                        "depth": {"value": 2.5, "unit": "not_a_unit"},
                    }
                ]
            }
        )
        with self.assertRaises(ValueError):
            get_matching_patient_cases(bundle)

    def test_discriminator_only_polymorphic_entries_emit_exists_sql(self):
        tnm_sql = str(
            get_matching_patient_cases(
                PartialPatientCaseBundle.model_validate(
                    {"stagings": [{"stagingDomain": "tnm"}]}
                )
            ).query
        )
        self.assertIn("EXISTS", tnm_sql)
        self.assertIn("oncology_tnmstaging", tnm_sql)

        mtb_sql = str(
            get_matching_patient_cases(
                PartialPatientCaseBundle.model_validate(
                    {"tumorBoards": [{"category": "molecular"}]}
                )
            ).query
        )
        self.assertIn("EXISTS", mtb_sql)
        self.assertIn("oncology_moleculartumorboard", mtb_sql)

    def test_non_polymorphic_category_column_is_filtered_not_ignored(self):
        sql = str(
            get_matching_patient_cases(
                PartialPatientCaseBundle.model_validate(
                    {
                        "adverseEvents": [
                            {"mitigations": [{"category": "pharmacological"}]}
                        ]
                    }
                )
            ).query
        )
        self.assertIn("oncology_adverseeventmitigation", sql)
        self.assertIn("pharmacological", sql)

    def test_non_integer_float_genomic_signature_excludes_integer_subtypes(self):
        sql = str(
            get_matching_patient_cases(
                PartialPatientCaseBundle.model_validate(
                    {"genomicSignatures": [{"value": 18.5}]}
                )
            ).query
        )
        self.assertIn("oncology_tumormutationalburden", sql)
        self.assertNotIn("oncology_aneuploidscore", sql)

    def test_empty_nested_child_entries_do_not_raise(self):
        sql = str(
            get_matching_patient_cases(
                PartialPatientCaseBundle.model_validate(
                    {"radiotherapies": [{"settings": [{}]}]}
                )
            ).query
        )
        self.assertNotIn("EXISTS", sql)

    def test_debug_logging_uses_parameterized_template_without_bound_values(self):
        secret_code = "DEBUG-SECRET-GENDER-999"
        bundle = PartialPatientCaseBundle.model_validate(
            {
                "gender": {"code": secret_code},
                "vitalStatus": PatientCaseVitalStatusChoices.ALIVE.value,
            }
        )
        with mock.patch.object(QuerySet, "count", return_value=3):
            _assert_debug_sql_template_logging(
                self,
                bundle,
                expected_count=3,
                forbidden_values=(
                    secret_code,
                    PatientCaseVitalStatusChoices.ALIVE.value,
                ),
            )


class TestMatchingEngine(TestCase):
    """Tests for ORM-based patient case matching across clinical and genomic panels."""

    def test_empty_bundle_counts_all_cases(self):
        PatientCaseFactory.create()
        PatientCaseFactory.create()
        self.assertEqual(_count({}), 2)

    def test_top_level_patient_case_attributes(self):
        c1 = PatientCaseFactory.create(
            vital_status=PatientCaseVitalStatusChoices.ALIVE,
            date_of_death=None,
            cause_of_death=None,
            end_of_records=None,
        )
        PatientCaseFactory.create(
            vital_status=PatientCaseVitalStatusChoices.DECEASED,
        )
        self.assertEqual(
            _count(
                {
                    "gender": {"code": c1.gender.code},
                    "vitalStatus": PatientCaseVitalStatusChoices.ALIVE.value,
                }
            ),
            1,
        )

    def test_neoplastic_entity_relationship_morphology_and_topography_group(self):
        sys_uri = "http://test.example/terminology/matching-neoplastic"
        m1, _ = terminology.CancerMorphology.objects.get_or_create(
            code="match-morph-1",
            system=sys_uri,
            defaults={"display": "Matching morphology 1"},
        )
        m2, _ = terminology.CancerMorphology.objects.get_or_create(
            code="match-morph-2",
            system=sys_uri,
            defaults={"display": "Matching morphology 2"},
        )
        entity = PrimaryNeoplasticEntityFactory.create(morphology=m1)
        PrimaryNeoplasticEntityFactory.create(morphology=m2)

        self.assertEqual(
            _count(
                {
                    "neoplasticEntities": [
                        {
                            "relationship": "primary",
                            "morphology": {"code": m1.code},
                            "topography": {"code": entity.topography.code},
                        }
                    ]
                }
            ),
            1,
        )

        # Verify SubqueryObjectProperty (topographyGroup) matching.
        if entity.topography_group is not None:
            self.assertGreaterEqual(
                _count(
                    {
                        "neoplasticEntities": [
                            {
                                "relationship": "primary",
                                "morphology": {"code": m1.code},
                                "topographyGroup": {
                                    "code": entity.topography_group.code
                                },
                            }
                        ]
                    }
                ),
                1,
            )

    def test_staging_with_and_without_domain_discriminator(self):
        tnm = TNMStagingFactory.create()
        lymphoma = LymphomaStagingFactory.create(bulky=True)

        # Explicit discriminator narrows to TNMStaging.
        self.assertGreaterEqual(
            _count(
                {
                    "stagings": [
                        {
                            "stagingDomain": "tnm",
                            "stage": {"code": tnm.stage.code},
                            "primaryTumor": {"code": tnm.primary_tumor.code},
                        }
                    ]
                }
            ),
            1,
        )

        # Omitting stagingDomain matches across compatible staging subclasses.
        qs = get_matching_patient_cases(
            PartialPatientCaseBundle.model_validate(
                {"stagings": [{"stage": {"code": tnm.stage.code}}]}
            )
        )
        self.assertIn(tnm.case_id, set(qs.values_list("pk", flat=True)))

        # Subclass-specific field without stagingDomain resolves to LymphomaStaging.
        qs_lymph = get_matching_patient_cases(
            PartialPatientCaseBundle.model_validate(
                {"stagings": [{"bulky": True, "stage": {"code": lymphoma.stage.code}}]}
            )
        )
        self.assertIn(lymphoma.case_id, set(qs_lymph.values_list("pk", flat=True)))
        self.assertNotIn(tnm.case_id, set(qs_lymph.values_list("pk", flat=True)))

        # Discriminator-only entry matches cases with that concrete staging subtype.
        qs_tnm_only = get_matching_patient_cases(
            PartialPatientCaseBundle.model_validate(
                {"stagings": [{"stagingDomain": "tnm"}]}
            )
        )
        self.assertIn(tnm.case_id, set(qs_tnm_only.values_list("pk", flat=True)))
        self.assertNotIn(
            lymphoma.case_id, set(qs_tnm_only.values_list("pk", flat=True))
        )

    def test_breslow_depth_staging_measurement_matching(self):
        breslow = BreslowDepthFactory.create(
            depth=measures.Distance(mm=2.5),
            is_ulcered=True,
        )
        BreslowDepthFactory.create(
            depth=measures.Distance(mm=0.5),
            is_ulcered=False,
        )
        qs = get_matching_patient_cases(
            PartialPatientCaseBundle.model_validate(
                {
                    "stagings": [
                        {
                            "stagingDomain": "breslow",
                            "depth": {"value": 2.5, "unit": "mm"},
                            "isUlcered": True,
                        }
                    ]
                }
            )
        )
        self.assertEqual(set(qs.values_list("pk", flat=True)), {breslow.case_id})

    def test_genomic_signature_polymorphic_matching(self):
        msi = MicrosatelliteInstabilityFactory.create()
        tmb = TumorMutationalBurdenFactory.create(
            status=TumorMutationalBurdenStatusChoices.HIGH,
            value=18.5,
        )

        # CodedConcept value without explicit category matches MicrosatelliteInstability.
        qs_msi = get_matching_patient_cases(
            PartialPatientCaseBundle.model_validate(
                {"genomicSignatures": [{"value": {"code": msi.value.code}}]}
            )
        )
        self.assertIn(msi.case_id, set(qs_msi.values_list("pk", flat=True)))

        # Explicit category + enum status matches TumorMutationalBurden.
        qs_tmb = get_matching_patient_cases(
            PartialPatientCaseBundle.model_validate(
                {
                    "genomicSignatures": [
                        {
                            "category": "tumor_mutational_burden",
                            "status": TumorMutationalBurdenStatusChoices.HIGH.value,
                        }
                    ]
                }
            )
        )
        self.assertIn(tmb.case_id, set(qs_tmb.values_list("pk", flat=True)))
        self.assertNotIn(msi.case_id, set(qs_tmb.values_list("pk", flat=True)))

    def test_tumor_board_and_molecular_therapeutic_recommendations(self):
        tb = TumorBoardFactory.create()
        rec_code = tb.recommendations.first().code

        mtb = MolecularTumorBoardFactory.create(conducted_cup_characterization=True)
        mol_rec = MolecularTherapeuticRecommendationFactory.create(
            molecular_tumor_board=mtb,
            off_label_use=True,
        )
        sys_uri = "http://test.example/terminology/matching-mtb-drug"
        drug, _ = terminology.AntineoplasticAgent.objects.get_or_create(
            code="MATCH-MTB-DRUG-1",
            system=sys_uri,
            defaults={"display": "Match MTB Drug 1"},
        )
        mol_rec.drugs.set([drug])

        # Shared parent field without category queries TumorBoard base directly.
        qs_tb = get_matching_patient_cases(
            PartialPatientCaseBundle.model_validate(
                {"tumorBoards": [{"recommendations": [{"code": rec_code}]}]}
            )
        )
        self.assertIn(tb.case_id, set(qs_tb.values_list("pk", flat=True)))

        # Nested therapeuticRecommendations matches MolecularTumorBoard.
        qs_mtb = get_matching_patient_cases(
            PartialPatientCaseBundle.model_validate(
                {
                    "tumorBoards": [
                        {
                            "conductedCupCharacterization": True,
                            "therapeuticRecommendations": [
                                {
                                    "offLabelUse": True,
                                    "drugs": [{"code": drug.code}],
                                }
                            ],
                        }
                    ]
                }
            )
        )
        self.assertEqual(set(qs_mtb.values_list("pk", flat=True)), {mtb.case_id})

    def test_performance_status_scores_and_interpretations(self):
        ps_ecog1 = ECOGPerformanceStatusFactory.create(ecog_score=1)
        ECOGPerformanceStatusFactory.create(ecog_score=3)
        ps_k80 = KarnofskyPerformanceStatusFactory.create(karnofsky_score=80)

        self.assertEqual(
            _count(
                {"performanceStatus": [{"ecogInterpretation": {"code": "LA9623-5"}}]}
            ),
            1,
        )
        self.assertEqual(
            set(
                get_matching_patient_cases(
                    PartialPatientCaseBundle.model_validate(
                        {"performanceStatus": [{"ecogScore": 1}]}
                    )
                ).values_list("pk", flat=True)
            ),
            {ps_ecog1.case_id},
        )
        self.assertEqual(
            set(
                get_matching_patient_cases(
                    PartialPatientCaseBundle.model_validate(
                        {
                            "performanceStatus": [
                                {"karnofskyInterpretation": {"code": "LA29177-5"}}
                            ]
                        }
                    )
                ).values_list("pk", flat=True)
            ),
            {ps_k80.case_id},
        )
        self.assertEqual(
            _count(
                {
                    "performanceStatus": [
                        {"ecogInterpretation": {"code": "UNKNOWN-LOINC"}}
                    ]
                }
            ),
            0,
        )

    def test_genomic_variant_fields_and_multi_gene_and(self):
        sys_uri = "http://test.example/terminology/matching-gene"
        g1, _ = terminology.Gene.objects.get_or_create(
            code="MATCH-GENE-A",
            system=sys_uri,
            defaults={
                "display": "Match gene A",
                "properties": {"location": "13q14.2"},
            },
        )
        g2, _ = terminology.Gene.objects.get_or_create(
            code="MATCH-GENE-B",
            system=sys_uri,
            defaults={
                "display": "Match gene B",
                "properties": {"location": "17p13.1"},
            },
        )
        protein_hgvs = "NP_003997.1:p.Trp24Cys"
        dna_hgvs = "NM_000123.1:c.100A>G"
        gv_match = GenomicVariantFactory.create(
            gene_panel="MSK-IMPACT-CNA",
            protein_hgvs=protein_hgvs,
            dna_hgvs=dna_hgvs,
        )
        gv_match.genes.set([g1, g2])

        gv_partial = GenomicVariantFactory.create(
            gene_panel="OTHER-PANEL",
            protein_hgvs="NP_003997.1:p.Trp24Ter",
            dna_hgvs="NM_000123.1:c.200T>C",
        )
        gv_partial.genes.set([g1])

        self.assertEqual(
            _count({"genomicVariants": [{"proteinHgvs": protein_hgvs}]}), 1
        )
        self.assertEqual(_count({"genomicVariants": [{"dnaHgvs": dna_hgvs}]}), 1)
        self.assertEqual(
            _count({"genomicVariants": [{"genePanel": "msk-impact-cna"}]}), 1
        )
        self.assertEqual(
            _count({"genomicVariants": [{"cytogeneticLocation": "13q14.2"}]}), 2
        )
        # Multi-gene AND: both g1 and g2 must be present on the same variant.
        self.assertEqual(
            set(
                get_matching_patient_cases(
                    PartialPatientCaseBundle.model_validate(
                        {
                            "genomicVariants": [
                                {"genes": [{"code": g1.code}, {"code": g2.code}]}
                            ]
                        }
                    )
                ).values_list("pk", flat=True)
            ),
            {gv_match.case_id},
        )

    def test_therapy_line_label_intent_ordinal(self):
        tl = TherapyLineFactory.create(
            intent=TherapyLineIntentChoices.PALLIATIVE, ordinal=2
        )
        tl.refresh_from_db()
        other = TherapyLineFactory.create(
            intent=TherapyLineIntentChoices.CURATIVE, ordinal=1
        )
        other.refresh_from_db()

        qs = get_matching_patient_cases(
            PartialPatientCaseBundle.model_validate(
                {
                    "therapyLines": [
                        {"label": tl.label, "intent": "palliative", "ordinal": 2}
                    ]
                }
            )
        )
        self.assertEqual(set(qs.values_list("pk", flat=True)), {tl.case_id})
        self.assertNotIn(other.case_id, set(qs.values_list("pk", flat=True)))

    def test_systemic_therapy_medications_and_semantics(self):
        sys_uri = "http://test.example/terminology/matching-drug"
        d1, _ = terminology.AntineoplasticAgent.objects.get_or_create(
            code="MATCH-DRUG-1",
            system=sys_uri,
            defaults={"display": "Match Drug 1"},
        )
        d2, _ = terminology.AntineoplasticAgent.objects.get_or_create(
            code="MATCH-DRUG-2",
            system=sys_uri,
            defaults={"display": "Match Drug 2"},
        )
        st_both = SystemicTherapyFactory.create(
            intent=SystemicTherapyIntentChoices.CURATIVE,
            medications=[],
        )
        SystemicTherapyMedicationFactory.create(systemic_therapy=st_both, drug=d1)
        SystemicTherapyMedicationFactory.create(systemic_therapy=st_both, drug=d2)

        st_one = SystemicTherapyFactory.create(
            intent=SystemicTherapyIntentChoices.CURATIVE,
            medications=[],
        )
        SystemicTherapyMedicationFactory.create(systemic_therapy=st_one, drug=d1)

        qs = get_matching_patient_cases(
            PartialPatientCaseBundle.model_validate(
                {
                    "systemicTherapies": [
                        {
                            "medications": [
                                {"drug": {"code": d1.code}},
                                {"drug": {"code": d2.code}},
                            ]
                        }
                    ]
                }
            )
        )
        self.assertEqual(set(qs.values_list("pk", flat=True)), {st_both.case_id})
        self.assertEqual(
            _count(
                {
                    "systemicTherapies": [
                        {"medications": [{"drug": {"code": "__nonexistent__"}}]}
                    ]
                }
            ),
            0,
        )

    def test_radiotherapy_nested_settings_same_row_constraint(self):
        sys_uri = "http://test.example/terminology/matching-rt"
        mod1, _ = terminology.RadiotherapyModality.objects.get_or_create(
            code="RT-MOD-1", system=sys_uri, defaults={"display": "Mod 1"}
        )
        mod2, _ = terminology.RadiotherapyModality.objects.get_or_create(
            code="RT-MOD-2", system=sys_uri, defaults={"display": "Mod 2"}
        )
        tech1, _ = terminology.RadiotherapyTechnique.objects.get_or_create(
            code="RT-TECH-1", system=sys_uri, defaults={"display": "Tech 1"}
        )
        tech2, _ = terminology.RadiotherapyTechnique.objects.get_or_create(
            code="RT-TECH-2", system=sys_uri, defaults={"display": "Tech 2"}
        )

        rt = RadiotherapyFactory.create(
            intent=RadiotherapyIntentChoices.CURATIVE,
            sessions=15,
            settings=[],
        )
        RadiotherapySettingFactory.create(
            radiotherapy=rt, modality=mod1, technique=tech1
        )
        RadiotherapySettingFactory.create(
            radiotherapy=rt, modality=mod2, technique=tech2
        )
        dosage = rt.dosages.first()

        # Matching modality + technique on the same setting row succeeds.
        self.assertEqual(
            _count(
                {
                    "radiotherapies": [
                        {
                            "intent": "curative",
                            "sessions": 15,
                            "dosages": [
                                {
                                    "irradiatedVolume": {
                                        "code": dosage.irradiated_volume.code
                                    }
                                }
                            ],
                            "settings": [
                                {
                                    "modality": {"code": mod1.code},
                                    "technique": {"code": tech1.code},
                                }
                            ],
                        }
                    ]
                }
            ),
            1,
        )

        # Cross-row combination (mod1 + tech2) must NOT match a single setting entry.
        self.assertEqual(
            _count(
                {
                    "radiotherapies": [
                        {
                            "settings": [
                                {
                                    "modality": {"code": mod1.code},
                                    "technique": {"code": tech2.code},
                                }
                            ]
                        }
                    ]
                }
            ),
            0,
        )

    def test_tumor_marker_and_surgery_and_multi_panel_and(self):
        case = PatientCaseFactory.create()
        tm = TumorMarkerTestFactory.create(
            case=case,
            mass_concentration=measures.MassConcentration(ng__ml=1.5),
        )
        surgery = SurgeryFactory.create(
            case=case,
            intent=SurgeryIntentChoices.CURATIVE,
        )
        TumorMarkerTestFactory.create(
            analyte=tm.analyte,
            mass_concentration=measures.MassConcentration(ng__ml=5.0),
        )

        self.assertEqual(
            _count(
                {
                    "tumorMarkers": [
                        {
                            "analyte": {"code": tm.analyte.code},
                            "massConcentration": {"value": 1.5, "unit": "ng__ml"},
                        }
                    ],
                    "surgeries": [
                        {
                            "procedure": {"code": surgery.procedure.code},
                            "intent": "curative",
                        }
                    ],
                }
            ),
            1,
        )

    def test_debug_logging_emits_sql_trace_only_in_debug_mode(self):
        secret_code = "DEBUG-SECRET-GENDER-999"
        gender, _ = terminology.AdministrativeGender.objects.get_or_create(
            code=secret_code,
            system="http://test.example/terminology/debug-log",
            defaults={"display": "Debug test gender"},
        )
        PatientCaseFactory.create(
            gender=gender,
            vital_status=PatientCaseVitalStatusChoices.ALIVE,
            date_of_death=None,
            cause_of_death=None,
            end_of_records=None,
        )
        bundle = PartialPatientCaseBundle.model_validate(
            {
                "gender": {"code": secret_code},
                "vitalStatus": PatientCaseVitalStatusChoices.ALIVE.value,
            }
        )
        _assert_debug_sql_template_logging(
            self,
            bundle,
            expected_count=1,
            forbidden_values=(
                secret_code,
                PatientCaseVitalStatusChoices.ALIVE.value,
            ),
        )


class TestMatchingController(ApiControllerTestMixin, TestCase):
    """HTTP integration tests for `POST /api/v1/matching/count`."""

    controller_path = "/api/v1/matching"

    @classmethod
    def setUpTestData(cls):
        super().setUpTestData()
        cls.entity = PrimaryNeoplasticEntityFactory.create()

    @parameterized.expand(GET_HTTP_SCENARIOS)
    def test_count_matching_cases_scenarios(self, scenario, config):
        response = self.call_api_endpoint(
            "POST",
            "/count",
            data={"bundle": {"neoplasticEntities": []}},
            **config,
        )
        if scenario == "HTTPS Authenticated":
            self.assertEqual(response.status_code, 200)
            result = response.json()
            self.assertIsInstance(result, int)
            self.assertEqual(result, 1)

    def test_count_matching_cases_unauthorized_returns_403(self):
        response = self.call_api_endpoint(
            "POST",
            "/count",
            expected_responses=(403,),
            authenticated=True,
            use_https=True,
            access_level=0,
            data={"bundle": {}},
        )
        self.assertEqual(response.status_code, 403)

    def test_count_matching_cases_validation_errors_return_422(self):
        for bad_payload in (
            {},
            {"bundle": {"notAPanel": []}},
            {"bundle": {"clinicalIdentifier": "SECRET-1"}},
            {"bundle": "{}"},
            {"bundle": {"stagings": [{"primaryTumor": {"code": "T2"}, "bulky": True}]}},
            {"bundle": {"genomicVariants": [{"hgvsVersion": 2}]}},
            {
                "bundle": {
                    "stagings": [
                        {
                            "stagingDomain": "breslow",
                            "depth": {"value": 2.0, "unit": "not_a_unit"},
                        }
                    ]
                }
            },
        ):
            response = self.call_api_endpoint(
                "POST",
                "/count",
                expected_responses=(422,),
                authenticated=True,
                use_https=True,
                access_level=4,
                data=bad_payload,
            )
            self.assertEqual(response.status_code, 422)


class TestMatchingOpenAPIDocumentation(SimpleTestCase):
    """Verifies OpenAPI endpoint contract and documentation tag ordering."""

    def test_matching_endpoint_spec_and_tab_order_after_cohorts(self):
        spec = api.get_openapi_schema()
        paths = spec["paths"]

        self.assertNotIn("/api/v1/patient-cases/match-count", paths)
        self.assertIn("/api/v1/matching/count", paths)

        op = paths["/api/v1/matching/count"]["post"]
        self.assertEqual(op["operationId"], "matchingCount")
        self.assertEqual(op["tags"], ["Matching"])
        ok_response = op["responses"].get(200) or op["responses"].get("200")
        self.assertIsNotNone(ok_response)
        self.assertEqual(
            ok_response["content"]["application/json"]["schema"]["type"],
            "integer",
        )
        unprocessable = op["responses"].get(422) or op["responses"].get("422")
        self.assertIsNotNone(unprocessable)

        tag_order: list[str] = []
        for path_item in paths.values():
            for operation in path_item.values():
                if isinstance(operation, dict):
                    for tag in operation.get("tags", []):
                        if tag not in tag_order:
                            tag_order.append(tag)

        self.assertIn("Cohorts", tag_order)
        self.assertIn("Matching", tag_order)
        self.assertEqual(
            tag_order.index("Matching"),
            tag_order.index("Cohorts") + 1,
            f"Expected 'Matching' tab immediately after 'Cohorts', got: {tag_order}",
        )
