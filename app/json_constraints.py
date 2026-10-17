"""Local JSON token filtering; it constrains format, never the chosen verdict."""

from copy import deepcopy


def bounded_schema(model_schema):
    schema = deepcopy(model_schema)

    def visit(node):
        if isinstance(node, dict):
            if node.get("type") == "string" and "enum" not in node:
                node["maxLength"] = min(node.get("maxLength", 240), 240)
            if node.get("type") == "array":
                node["maxItems"] = max(
                    node.get("minItems", 0), min(node.get("maxItems", 6), 6)
                )
            if node.get("type") == "object" and "properties" in node:
                node["required"] = list(node["properties"])
            for child in node.values():
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)

    visit(schema)
    return schema


def json_prefix(tokenizer_data, model_schema):
    from lmformatenforcer import CharacterLevelParserConfig, JsonSchemaParser
    from lmformatenforcer.integrations.transformers import (
        build_transformers_prefix_allowed_tokens_fn,
    )

    config = CharacterLevelParserConfig(
        # 0.11.3 caches the alphabet used before minLength is reached at
        # construction time. It must already include the Qwen Unicode alphabet.
        alphabet=tokenizer_data.tokenizer_alphabet,
        force_json_field_order=True, max_consecutive_whitespaces=2,
    )
    parser = JsonSchemaParser(bounded_schema(model_schema), config=config)
    prefix = build_transformers_prefix_allowed_tokens_fn(tokenizer_data, parser)
    # TokenEnforcer replaces the config; preserve our order/whitespace limits.
    parser.config = config
    return prefix
