import json

import pytest

from app.json_constraints import bounded_schema
from app.screening import ScreeningReport
from app.schemas import Category


def consume(parser, text):
    for char in text:
        if char not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(char)
    return parser.can_end()


def test_bounds_keep_all_seven_categories_and_original_schema_unchanged():
    original = ScreeningReport.model_json_schema()
    bounded = bounded_schema(original)
    assert bounded["properties"]["category_checks"]["maxItems"] == 7
    assert bounded["properties"]["observations"]["maxItems"] == 6
    assert original["properties"]["observations"]["maxItems"] == 12
    assert bounded["$defs"]["VisualCheck"]["properties"]["evidence_frame_indices"]["maxItems"] == 3


def test_real_json_parser_accepts_negative_and_positive_but_bounds_evidence():
    lmfe = pytest.importorskip("lmformatenforcer")
    schema = {
        "type": "object", "properties": {
            "seen": {"type": "boolean"},
            "evidence": {"type": "array", "items": {"type": "integer"}, "maxItems": 3},
        }, "required": ["seen", "evidence"], "additionalProperties": False,
    }
    assert consume(lmfe.JsonSchemaParser(schema), json.dumps({"seen": False, "evidence": []}))
    assert consume(lmfe.JsonSchemaParser(schema), json.dumps({"seen": True, "evidence": [0, 2]}))
    assert not consume(lmfe.JsonSchemaParser(schema), json.dumps({"seen": True, "evidence": [0, 1, 2, 3]}))


def test_full_screening_schema_refs_nulls_and_positive_trigger_are_supported():
    lmfe = pytest.importorskip("lmformatenforcer")
    data = {
        "scene_summary": "Synthetic local test only",
        "observations": ["Synthetic action for parser compatibility"],
        "category_checks": [
            {"category": category.value, "visual_trigger_seen": False,
             "start_frame_index": None, "end_frame_index": None,
             "observed_action": "No visible action in synthetic test",
             "uncertainty_reason": "Synthetic input", "evidence_frame_indices": []}
            for category in Category
        ], "limitations": [], "review_notes": [],
    }
    schema = bounded_schema(ScreeningReport.model_json_schema())
    config = lmfe.CharacterLevelParserConfig(force_json_field_order=True)
    assert consume(lmfe.JsonSchemaParser(schema, config), json.dumps(data))
    data["category_checks"][0].update(visual_trigger_seen=True, start_frame_index=0,
        end_frame_index=2, evidence_frame_indices=[1])
    assert consume(lmfe.JsonSchemaParser(schema, config), json.dumps(data))
