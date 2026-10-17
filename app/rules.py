import json
from pathlib import Path

RULES = json.loads(Path(__file__).with_name("rules.json").read_text())
RULE_BY_ID = {r["id"]: r for r in RULES}


def source_citation(rule):
    """Human-readable citation, or an honest note when no пункт is verified."""
    source = rule.get("normative_source")
    if not source or rule.get("source_verification_status") != "verified":
        return "Нормативный пункт для этой категории не подтверждён."
    section = f", {source['section']}" if source.get("section") else ""
    return f"Источник: {source['title']}{section}."


def violation_description(event):
    rule = RULE_BY_ID[event.category.value]
    parts = [
        event.observed_action.strip(),
        f"Почему это может быть нарушением: {event.applicability_basis.strip()}",
        f"Правило: {rule['name']}. {source_citation(rule)}",
        f"Ограничения вывода: {event.uncertainty_reason.strip()}",
    ]
    if event.category.value == "unregistered_cash_transfer_between_workers":
        parts.append("Регистрация передачи по видео не проверяется.")
    return " ".join(parts)
