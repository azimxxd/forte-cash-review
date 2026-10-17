import json
from pathlib import Path

RULES = json.loads(Path(__file__).with_name("rules.json").read_text())
RULE_BY_ID = {r["id"]: r for r in RULES}
