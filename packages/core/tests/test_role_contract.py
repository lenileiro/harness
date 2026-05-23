"""Tests for the RoleContract typed-contract layer."""

from __future__ import annotations

from harness.core import (
    Authority,
    RoleContract,
    filter_tools_by_authority,
    validate_inputs,
    validate_outputs,
)


class TestAuthorityFiltering:
    def test_empty_authority_returns_all_tools(self) -> None:
        contract = RoleContract(role="worker")
        result = filter_tools_by_authority(["a", "b", "c"], contract)
        assert result == ["a", "b", "c"]

    def test_filters_to_intersection_preserving_order(self) -> None:
        contract = RoleContract(
            role="worker",
            authority=Authority(allowed_tools=("c", "a")),
        )
        result = filter_tools_by_authority(["a", "b", "c"], contract)
        assert result == ["a", "c"]

    def test_unknown_tools_in_authority_silently_dropped(self) -> None:
        contract = RoleContract(
            role="worker",
            authority=Authority(allowed_tools=("a", "missing")),
        )
        result = filter_tools_by_authority(["a", "b"], contract)
        assert result == ["a"]


class TestInputValidation:
    def test_empty_schema_passes_anything(self) -> None:
        contract = RoleContract(role="worker")
        assert validate_inputs(contract, {"random": 1})

    def test_missing_required_key_fails(self) -> None:
        contract = RoleContract(
            role="worker",
            inputs_schema={
                "type": "object",
                "required": ["task"],
                "properties": {"task": {"type": "string"}},
            },
        )
        out = validate_inputs(contract, {})
        assert not out
        assert any("missing" in i for i in out.issues)

    def test_wrong_type_fails(self) -> None:
        contract = RoleContract(
            role="worker",
            inputs_schema={
                "type": "object",
                "properties": {"count": {"type": "integer"}},
            },
        )
        out = validate_inputs(contract, {"count": "not-an-int"})
        assert not out


class TestOutputValidation:
    def test_text_with_no_schema_passes(self) -> None:
        contract = RoleContract(role="reporter")
        assert validate_outputs(contract, "free-form text")

    def test_json_string_parsed_against_schema(self) -> None:
        contract = RoleContract(
            role="reporter",
            outputs_schema={
                "type": "object",
                "required": ["summary"],
                "properties": {"summary": {"type": "string"}},
            },
        )
        assert validate_outputs(contract, '{"summary": "ok"}')

    def test_text_when_schema_expects_object_fails(self) -> None:
        contract = RoleContract(
            role="reporter",
            outputs_schema={"type": "object", "required": ["summary"]},
        )
        out = validate_outputs(contract, "just text")
        assert not out


class TestSerialization:
    def test_as_dict_roundtrip(self) -> None:
        contract = RoleContract(
            role="planner",
            inputs_schema={"type": "object"},
            outputs_schema={"type": "object", "required": ["plan"]},
            authority=Authority(allowed_tools=("read_file",), cwd_subpaths=("src/",)),
            stop_condition="all work items created",
            max_turns=5,
        )
        data = contract.as_dict()
        assert data["role"] == "planner"
        assert data["authority"]["allowed_tools"] == ["read_file"]
        assert data["max_turns"] == 5
