import json

from django.test import Client, TestCase

import onconova.terminology.models as terminology
from onconova.oncology.similarity_count import (
    _FUNCTIONAL_AGGREGATION_PANEL_KEYS,
    count_patient_cases_matching_functional_aggregated_data,
)
from onconova.tests.factories import (
    ECOGPerformanceStatusFactory,
    PrimaryNeoplasticEntityFactory,
    UserFactory,
)


class TestFunctionalAggregatedDataCount(TestCase):
    def test_panel_keys_cover_aitb_dashboard_panels(self):
        """Keeps server in sync with aitb-dashboard ``casePanelColumns`` / ``BUNDLE_ARRAY_KEYS_ORDER``."""
        expected = frozenset(
            {
                "neoplasticEntities",
                "stagings",
                "tumorMarkers",
                "riskAssessments",
                "genomicVariants",
                "genomicSignatures",
                "therapyLines",
                "systemicTherapies",
                "surgeries",
                "radiotherapies",
                "adverseEvents",
                "treatmentResponses",
                "performanceStatus",
                "comorbidities",
                "vitals",
                "lifestyles",
                "familyHistory",
                "tumorBoards",
            }
        )
        self.assertEqual(_FUNCTIONAL_AGGREGATION_PANEL_KEYS, expected)

    def test_empty_bundle_counts_all_cases(self):
        PrimaryNeoplasticEntityFactory.create()
        PrimaryNeoplasticEntityFactory.create()
        n = count_patient_cases_matching_functional_aggregated_data({})
        self.assertEqual(n, 2)

    def test_case_example_wrapper_is_normalized(self):
        PrimaryNeoplasticEntityFactory.create()
        PrimaryNeoplasticEntityFactory.create()
        n = count_patient_cases_matching_functional_aggregated_data({"caseExample": {}})
        self.assertEqual(n, 2)

    def test_unknown_panel_key_raises(self):
        with self.assertRaises(ValueError) as ctx:
            count_patient_cases_matching_functional_aggregated_data(
                {"notAPanel": [{"x": {"code": "1"}}]}
            )
        self.assertIn("notAPanel", str(ctx.exception))

    def test_neoplastic_morphology_and_relationship(self):
        sys_uri = "http://test.example/terminology/functional-agg-count-neoplastic"
        m1, _ = terminology.CancerMorphology.objects.get_or_create(
            code="fat-neo-morph-1",
            system=sys_uri,
            defaults={"display": "Functional agg test morphology 1"},
        )
        m2, _ = terminology.CancerMorphology.objects.get_or_create(
            code="fat-neo-morph-2",
            system=sys_uri,
            defaults={"display": "Functional agg test morphology 2"},
        )
        entity = PrimaryNeoplasticEntityFactory.create(morphology=m1)
        PrimaryNeoplasticEntityFactory.create(morphology=m2)

        payload = {
            "neoplasticEntities": [
                {
                    "relationship": {"code": "primary"},
                    "morphology": {"code": m1.code},
                }
            ]
        }
        n = count_patient_cases_matching_functional_aggregated_data(payload)
        self.assertEqual(n, 1)
        self.assertEqual(entity.case_id, entity.case.id)

    def test_performance_status_ecog_interpretation_in_sql_and_count(self):
        from onconova.oncology.similarity_count import (
            interpolated_sql_for_functional_aggregated_patient_case_filter,
            patient_cases_queryset_for_functional_aggregated_data,
        )

        ps_ecog1 = ECOGPerformanceStatusFactory.create(ecog_score=1)
        ECOGPerformanceStatusFactory.create(ecog_score=3)
        payload = {
            "performanceStatus": [
                {"ecogInterpretation": {"code": "LA9623-5"}},
            ],
        }
        n = count_patient_cases_matching_functional_aggregated_data(payload)
        self.assertEqual(n, 1)
        self.assertEqual(ps_ecog1.case_id, ps_ecog1.case.id)

        pqs = patient_cases_queryset_for_functional_aggregated_data(payload)
        sql = interpolated_sql_for_functional_aggregated_patient_case_filter(pqs).lower()
        self.assertIn("oncology_performancestatus", sql)
        self.assertIn("ecog_score", sql)

    def test_validate_endpoint_post_json_body(self):
        user = UserFactory.create(access_level=4)
        user.set_password("test-pass-123")
        user.save()
        client = Client()
        auth = client.post(
            "/api/v1/auth/session",
            data={"username": user.username, "password": "test-pass-123"},
            content_type="application/json",
            secure=True,
        )
        self.assertEqual(auth.status_code, 200)
        token = auth.json()["sessionToken"]
        headers = {"X-Session-Token": str(token)}

        body = {"caseExample": {"neoplasticEntities": []}}
        r = Client(headers=headers).post(
            "/api/v1/patient-cases/similarity-count",
            data=json.dumps(body),
            content_type="application/json",
            secure=True,
        )
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("patientCaseCount", data)
        self.assertIn("patientCountSql", data)
        self.assertIsInstance(data["patientCountSql"], str)
        self.assertGreater(len(data["patientCountSql"]), 0)

        r_string_form = Client(headers=headers).post(
            "/api/v1/patient-cases/similarity-count",
            data=json.dumps(
                {
                    "caseExample": json.dumps({"neoplasticEntities": []})
                }
            ),
            content_type="application/json",
            secure=True,
        )
        self.assertEqual(r_string_form.status_code, 200)

        r2 = Client(headers=headers).post(
            "/api/v1/patient-cases/similarity-count",
            data=json.dumps({}),
            content_type="application/json",
            secure=True,
        )
        self.assertEqual(r2.status_code, 422)

    def test_genomic_variant_row_q_includes_gene_panel_in_sql(self):
        from onconova.oncology import models as orm
        from onconova.oncology.similarity_count import (
            _genomic_variant_row_q,
            interpolated_sql_for_functional_aggregated_patient_case_filter,
            patient_cases_queryset_for_functional_aggregated_data,
        )

        q = _genomic_variant_row_q({"genePanel": {"code": "MSK-IMPACT-CNA"}})
        self.assertIsNotNone(q)
        qs = orm.PatientCase.objects.all().filter(q)
        compiler = qs.query.get_compiler(using=qs.db)
        sql, _params = compiler.as_sql()
        lowered = sql.lower()
        self.assertIn("oncology_genomicvariant", lowered)
        self.assertIn("gene_panel", lowered)

        payload = {
            "genomicVariants": [
                {
                    "genePanel": {"code": "MSK-IMPACT-CNA"},
                    "cytogeneticLocation": {"code": "13q14.2"},
                }
            ]
        }
        pqs = patient_cases_queryset_for_functional_aggregated_data(payload)
        full = interpolated_sql_for_functional_aggregated_patient_case_filter(pqs).lower()
        self.assertIn("oncology_genomicvariant", full)
