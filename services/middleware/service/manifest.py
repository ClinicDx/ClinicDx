"""Manifest generation for Voice Scribe.

Queries the live OpenMRS instance to build a stripped-down concept manifest
for a given encounter type. The manifest is injected into the model context
at inference time so the model selects only from what's available.
"""

import json
import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

OPENMRS_BASE = os.environ.get("OPENMRS_URL", "http://localhost:8080/openmrs")
OPENMRS_USER = os.environ.get("OPENMRS_USER", "admin")
OPENMRS_PASS = os.environ.get("OPENMRS_PASSWORD", "Admin123")

# Maps encounter type UUID → list of CIEL concept codes to include in manifest.
# Derived from the live OpenMRS instance observation data.
ENCOUNTER_CONCEPT_MAP: dict[str, list[str]] = {
    # Vitals encounter
    "67a71486-1a54-468f-ac3e-7091a9a79584": [
        "5088", "5085", "5086", "5087", "5092", "5089", "5090", "5242",
        "1342", "5314",
    ],
    # Visit Note / Consultation
    "d7151f82-c1f3-4152-a605-2f9ea7414a79": [
        "5088", "5085", "5086", "5087", "5092", "5089", "5090", "5242",
        "1342", "159368", "160347", "166589", "161422",
        "730", "5497", "160912", "169187", "167176", "887",
    ],
    "dd528487-82a5-4082-9c72-ed246bd49591": [
        "5088", "5085", "5086", "5087", "5092", "5089", "5090", "5242",
        "1342", "159368", "160347", "166589", "161422",
        "730", "5497", "160912", "169187", "167176", "887",
    ],
    # Lab Results
    "3596fafb-6f6f-4396-8c87-6e63a0f1bd71": [
        "730", "5497", "160912", "169187", "167176", "887",
        "162660", "5475",
    ],
    # Adult Visit
    "0e8230ce-bd1d-43f5-a863-cf44344fa4b0": [
        "5088", "5085", "5086", "5087", "5092", "5089", "5090", "5242",
        "1342", "159368", "160347", "166589", "161422",
        "730", "5497", "160912",
    ],
    # Antenatal / Obstetric visit
    "e22e39fd-7db2-45e7-80f1-60fa0d5a4378": [
        "5088", "5085", "5086", "5087", "5092", "5089", "5090", "5242",
        "163166", "1160", "1439", "163460", "5916",
        "165380", "166604",
    ],
    # Pediatric / Child visit
    "6d88e570-a489-4e31-9e0a-5406c2e75d9e": [
        "5088", "5085", "5086", "5087", "5092", "5089", "5090", "5242",
        "5314", "1343", "5916", "166604",
    ],
}

# Fallback concept set for unknown encounter types — all trained concepts
DEFAULT_CONCEPT_CODES = [
    "5088", "5085", "5086", "5087", "5092", "5089", "5090", "5242",
    "1342", "159368", "160347", "166589", "161422",
    "730", "5497", "160912", "169187", "167176", "887",
    "163166", "1160", "1439", "163460", "5916", "165380", "166604",
    "5314", "1343", "162660",
]

# FHIR resource type per CIEL concept category
CONCEPT_FHIR_TYPE = {
    "exam": "Observation",
    "laboratory": "Observation",
    "condition": "Condition",
}

# Labels must match training data exactly (clips.jsonl manifest_line field).
# The model was trained on these exact strings; any deviation breaks extraction.
# Full 29-concept set derived from clips.jsonl concepts field.
CIEL_LABELS: dict[str, dict] = {
    # ── Vitals ────────────────────────────────────────────────────────────────
    "5088":   {"label": "temperature_c",                               "manifest_line": "temperature_c (number, C)",                               "unit": "C",           "fhir_type": "Observation", "category": "exam",       "value_type": "Quantity"},
    "5085":   {"label": "systolic_blood_pressure",                     "manifest_line": "systolic_blood_pressure (number, mmHg)",                  "unit": "mmHg",        "fhir_type": "Observation", "category": "exam",       "value_type": "Quantity"},
    "5086":   {"label": "diastolic_blood_pressure",                    "manifest_line": "diastolic_blood_pressure (number, mmHg)",                 "unit": "mmHg",        "fhir_type": "Observation", "category": "exam",       "value_type": "Quantity"},
    "5087":   {"label": "pulse",                                       "manifest_line": "pulse (number, beats/min)",                               "unit": "beats/min",   "fhir_type": "Observation", "category": "exam",       "value_type": "Quantity"},
    "5092":   {"label": "arterial_blood_oxygen_saturation_pulse_oximeter", "manifest_line": "arterial_blood_oxygen_saturation_pulse_oximeter (number)", "unit": "%",      "fhir_type": "Observation", "category": "exam",       "value_type": "Quantity"},
    "5089":   {"label": "weight_kg",                                   "manifest_line": "weight_kg (number, kg)",                                  "unit": "kg",          "fhir_type": "Observation", "category": "exam",       "value_type": "Quantity"},
    "5090":   {"label": "height_cm",                                   "manifest_line": "height_cm (number, cm)",                                  "unit": "cm",          "fhir_type": "Observation", "category": "exam",       "value_type": "Quantity"},
    "5242":   {"label": "respiratory_rate",                            "manifest_line": "respiratory_rate (number)",                               "unit": "breaths/min", "fhir_type": "Observation", "category": "exam",       "value_type": "Quantity"},
    "1342":   {"label": "body_mass_index",                             "manifest_line": "body_mass_index (number, kg/m2)",                         "unit": "kg/m2",       "fhir_type": "Observation", "category": "exam",       "value_type": "Quantity"},
    # ── General clinical ──────────────────────────────────────────────────────
    "159368": {"label": "duration_of_illness",                         "manifest_line": "duration_of_illness (number)",                            "unit": None,          "fhir_type": "Observation", "category": "exam",       "value_type": "Quantity"},
    "160347": {"label": "glasgow_coma_scale",                          "manifest_line": "glasgow_coma_scale (number)",                             "unit": None,          "fhir_type": "Observation", "category": "exam",       "value_type": "Quantity"},
    "166589": {"label": "visual_analog_scale_pain_score",              "manifest_line": "visual_analog_scale_pain_score (number)",                 "unit": None,          "fhir_type": "Observation", "category": "exam",       "value_type": "Quantity"},
    "161422": {"label": "number_of_missed_medication_doses_yesterday", "manifest_line": "number_of_missed_medication_doses_yesterday (number)",    "unit": None,          "fhir_type": "Observation", "category": "exam",       "value_type": "Quantity"},
    "162660": {"label": "total_urine_output_over_24_hours_ml",         "manifest_line": "total_urine_output_over_24_hours_ml (number, mL)",        "unit": "mL",          "fhir_type": "Observation", "category": "exam",       "value_type": "Quantity"},
    # ── Lab / Tests ───────────────────────────────────────────────────────────
    "730":    {"label": "cd4_percent",                                 "manifest_line": "[test] cd4_percent (number)",                             "unit": "%",           "fhir_type": "Observation", "category": "laboratory", "value_type": "Quantity"},
    "5497":   {"label": "cd4_count",                                   "manifest_line": "[test] cd4_count (number)",                               "unit": "cells/uL",    "fhir_type": "Observation", "category": "laboratory", "value_type": "Quantity"},
    "160912": {"label": "fasting_blood_glucose_measurement_mg_dl",     "manifest_line": "[test] fasting_blood_glucose_measurement_mg_dl",          "unit": "mg/dL",       "fhir_type": "Observation", "category": "laboratory", "value_type": "Quantity"},
    "169187": {"label": "finger_stick_blood_glucose",                  "manifest_line": "[test] finger_stick_blood_glucose",                       "unit": "mg/dL",       "fhir_type": "Observation", "category": "laboratory", "value_type": "Quantity"},
    "167176": {"label": "post_prandial_blood_glucose_measurement_mg_dl_1_5_hour_after_meal", "manifest_line": "[test] post_prandial_blood_glucose_measurement_mg_dl_1_5_hour_after_meal", "unit": "mg/dL", "fhir_type": "Observation", "category": "laboratory", "value_type": "Quantity"},
    "887":    {"label": "serum_glucose",                               "manifest_line": "[test] serum_glucose",                                    "unit": "mg/dL",       "fhir_type": "Observation", "category": "laboratory", "value_type": "Quantity"},
    "5475":   {"label": "tuberculin_skin_test_mm",                     "manifest_line": "[test] tuberculin_skin_test_mm",                          "unit": "mm",          "fhir_type": "Observation", "category": "laboratory", "value_type": "Quantity"},
    # ── Pediatric / Anthropometric ────────────────────────────────────────────
    "5314":   {"label": "head_circumference",                          "manifest_line": "head_circumference (number, cm)",                         "unit": "cm",          "fhir_type": "Observation", "category": "exam",       "value_type": "Quantity"},
    "1343":   {"label": "mid_upper_arm_circumference_cm",              "manifest_line": "mid_upper_arm_circumference_cm (number)",                 "unit": "cm",          "fhir_type": "Observation", "category": "exam",       "value_type": "Quantity"},
    "5916":   {"label": "birth_weight_kg",                             "manifest_line": "birth_weight_kg (number, kg)",                            "unit": "kg",          "fhir_type": "Observation", "category": "exam",       "value_type": "Quantity"},
    "166604": {"label": "weight_on_admission_kg",                      "manifest_line": "weight_on_admission_kg (number, kg)",                     "unit": "kg",          "fhir_type": "Observation", "category": "exam",       "value_type": "Quantity"},
    # ── Obstetric ─────────────────────────────────────────────────────────────
    "163166": {"label": "estimated_gestational_age_weeks",             "manifest_line": "estimated_gestational_age_weeks (number)",                "unit": "weeks",       "fhir_type": "Observation", "category": "exam",       "value_type": "Quantity"},
    "1160":   {"label": "fetal_heart_rate",                            "manifest_line": "fetal_heart_rate (number)",                               "unit": "beats/min",   "fhir_type": "Observation", "category": "exam",       "value_type": "Quantity"},
    "1439":   {"label": "fundal_height",                               "manifest_line": "fundal_height (number, cm)",                              "unit": "cm",          "fhir_type": "Observation", "category": "exam",       "value_type": "Quantity"},
    "163460": {"label": "pre_gestational_weight_kg",                   "manifest_line": "pre_gestational_weight_kg (number, kg)",                  "unit": "kg",          "fhir_type": "Observation", "category": "exam",       "value_type": "Quantity"},
    "165380": {"label": "weight_gain_since_last_visit_kg",             "manifest_line": "weight_gain_since_last_visit_kg (number, kg)",            "unit": "kg",          "fhir_type": "Observation", "category": "exam",       "value_type": "Quantity"},
}


class ManifestBuilder:
    """Builds concept manifests from the live OpenMRS instance."""

    def __init__(self):
        self._auth = (OPENMRS_USER, OPENMRS_PASS)
        self._concept_cache: dict[str, dict] = {}

    async def get_encounter_context(self, encounter_uuid: str) -> dict:
        """Fetch encounter metadata: patient_uuid, encounter_type_uuid, location."""
        url = f"{OPENMRS_BASE}/ws/rest/v1/encounter/{encounter_uuid}"
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True, verify=False) as client:
            resp = await client.get(url, auth=self._auth)
            resp.raise_for_status()
            data = resp.json()
            return {
                "encounter_uuid": encounter_uuid,
                "patient_uuid": (data.get("patient") or {}).get("uuid"),
                "encounter_type_uuid": (data.get("encounterType") or {}).get("uuid"),
                "encounter_type_name": (data.get("encounterType") or {}).get("display"),
                "location_uuid": (data.get("location") or {}).get("uuid"),
                "location_name": (data.get("location") or {}).get("display"),
                "provider_uuid": (
                    (data.get("encounterProviders", [{}])[0].get("provider") or {}).get("uuid")
                    if data.get("encounterProviders")
                    else None
                ),
            }

    async def resolve_concept_uuid(self, ciel_code: str) -> Optional[str]:
        """Resolve a CIEL code to a local OpenMRS concept UUID."""
        if ciel_code in self._concept_cache:
            return self._concept_cache[ciel_code].get("uuid")

        url = f"{OPENMRS_BASE}/ws/rest/v1/concept?source=CIEL&code={ciel_code}"
        try:
            async with httpx.AsyncClient(timeout=8.0, follow_redirects=True, verify=False) as client:
                resp = await client.get(url, auth=self._auth)
                resp.raise_for_status()
                results = resp.json().get("results", [])
                if results:
                    uuid = results[0].get("uuid")
                    self._concept_cache[ciel_code] = {"uuid": uuid}
                    return uuid
        except Exception as e:
            logger.warning("Could not resolve CIEL:%s → %s", ciel_code, e)
        return None

    async def build_manifest(self, encounter_uuid: str) -> dict:
        """Build the full manifest for an encounter.

        Returns:
            {
                "context": { encounter/patient/location metadata },
                "manifest_string": "AVAILABLE:\ntemperature (number, celsius)\n...",
                "lookup": { "temperature": { ciel, uuid, fhir_type, unit, ... } }
            }
        """
        context = await self.get_encounter_context(encounter_uuid)
        encounter_type_uuid = context.get("encounter_type_uuid", "")

        ciel_codes = ENCOUNTER_CONCEPT_MAP.get(encounter_type_uuid, DEFAULT_CONCEPT_CODES)

        manifest_lines: list[str] = ["CONCEPTS:"]
        lookup: dict[str, dict] = {}

        for code in ciel_codes:
            meta = CIEL_LABELS.get(code)
            if not meta:
                continue

            label = meta["label"]
            manifest_line = meta.get("manifest_line", label)
            unit = meta["unit"]
            value_type = meta["value_type"]
            fhir_type = meta["fhir_type"]
            category = meta["category"]

            local_uuid = await self.resolve_concept_uuid(code)
            ciel_uuid_full = f"{code}AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"[:36]

            manifest_lines.append(manifest_line)

            lookup[label] = {
                "ciel_code": code,
                "ciel_uuid_full": ciel_uuid_full,
                "local_uuid": local_uuid or ciel_uuid_full,
                "fhir_type": fhir_type,
                "category": category,
                "unit": unit,
                "value_type": value_type,
                "display_name": label.replace("_", " ").title(),
            }

        return {
            "context": context,
            "manifest_string": "\n".join(manifest_lines),
            "lookup": lookup,
        }


# Singleton
_builder: Optional[ManifestBuilder] = None


def get_builder() -> ManifestBuilder:
    global _builder
    if _builder is None:
        _builder = ManifestBuilder()
    return _builder
