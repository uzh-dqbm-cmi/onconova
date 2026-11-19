import json

from django.test import Client, TestCase

import onconova.terminology.models as terminology
from onconova.oncology.match_explorer import (
    remove_property_paths_from_bundle,
    run_match_explorer,
)
from onconova.tests.factories import PrimaryNeoplasticEntityFactory, UserFactory


class TestMatchExplorerPathRemoval(TestCase):
    def test_remove_nested_leaf_prunes_row(self):
        bundle = {
            "neoplasticEntities": [
                {
                    "relationship": {"code": "primary"},
                    "morphology": {"code": "m1"},
                }
            ]
        }
        remove_property_paths_from_bundle(
            bundle, ["neoplasticEntities[0].morphology.code"]
        )
        self.assertEqual(bundle["neoplasticEntities"], [{"relationship": {"code": "primary"}}])

    def test_remove_all_row_fields_drops_row(self):
        bundle = {
            "neoplasticEntities": [
                {"relationship": {"code": "primary"}, "morphology": {"code": "m1"}}
            ]
        }
        remove_property_paths_from_bundle(
            bundle,
            [
                "neoplasticEntities[0].relationship.code",
                "neoplasticEntities[0].morphology.code",
            ],
        )
        self.assertEqual(bundle.get("neoplasticEntities"), [])


class TestMatchExplorerAPI(TestCase):
    def test_explorer_endpoint_returns_baseline_and_options(self):
        sys_uri = "http://test.example/terminology/functional-agg-explorer"
        m1, _ = terminology.CancerMorphology.objects.get_or_create(
            code="fat-neo-explorer-1",
            system=sys_uri,
            defaults={"display": "explorer morphology 1"},
        )
        m2, _ = terminology.CancerMorphology.objects.get_or_create(
            code="fat-neo-explorer-2",
            system=sys_uri,
            defaults={"display": "explorer morphology 2"},
        )
        PrimaryNeoplasticEntityFactory.create(morphology=m1)
        PrimaryNeoplasticEntityFactory.create(morphology=m2)

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

        case_example = {
            "neoplasticEntities": [
                {
                    "relationship": {"code": "primary"},
                    "morphology": {"code": m1.code},
                }
            ]
        }
        body = {
            "caseExample": case_example,
            "caseSelectorOptions": [
                {
                    "id": 0,
                    "path": "neoplasticEntities/0",
                    "properties": ["neoplasticEntities[0].morphology.code"],
                },
                {"id": 1, "path": "neoplasticEntities/0", "properties": []},
            ],
        }
        r = Client(headers=headers).post(
            "/api/v1/patient-cases/match-explorer",
            data=json.dumps(body),
            content_type="application/json",
            secure=True,
        )
        self.assertEqual(r.status_code, 200, r.content)
        data = r.json()
        self.assertIn("patientCaseCount", data)
        self.assertEqual(data["patientCaseCount"], 1)
        self.assertIn("patientCountSql", data)
        self.assertIn("caseSelectorOptions", data)
        opts = data["caseSelectorOptions"]
        self.assertEqual(len(opts), 2)
        self.assertEqual(opts[0]["path"], "neoplasticEntities/0")
        self.assertGreaterEqual(opts[0]["patientCaseCountIncrease"], 0)
        self.assertEqual(opts[1]["patientCaseCountIncrease"], 0)

    def test_run_match_explorer_empty_options(self):
        out = run_match_explorer({"neoplasticEntities": []}, [])
        self.assertIn("patientCaseCount", out)
        self.assertEqual(out["caseSelectorOptions"], [])
