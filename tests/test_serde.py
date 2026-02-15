import math
from collections import UserDict
from types import MappingProxyType

import pytest

from lmctx.serde import (
    as_str_object_dict,
    optional_float,
    optional_int,
    optional_string,
    require_int,
    string_tuple,
    to_plain_data,
)

# =============================================================================
# to_plain_data
# =============================================================================


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(42, 42, id="int"),
        pytest.param("hello", "hello", id="string"),
        pytest.param(None, None, id="none"),
        pytest.param(3.14, 3.14, id="float"),
        pytest.param(True, True, id="bool"),
    ],
)
def test_to_plain_data_passes_through_scalars(value: object, expected: object) -> None:
    assert to_plain_data(value) == expected


def test_to_plain_data_converts_mapping_to_dict() -> None:
    proxy = MappingProxyType({"a": 1, "b": 2})
    result = to_plain_data(proxy)
    assert result == {"a": 1, "b": 2}
    assert type(result) is dict


def test_to_plain_data_converts_tuple_to_list() -> None:
    result = to_plain_data((1, 2, 3))
    assert result == [1, 2, 3]
    assert type(result) is list


def test_to_plain_data_converts_nested_structures() -> None:
    nested = MappingProxyType(
        {
            "items": (MappingProxyType({"key": "value"}),),
        }
    )
    assert to_plain_data(nested) == {"items": [{"key": "value"}]}


def test_to_plain_data_converts_list_recursively() -> None:
    assert to_plain_data([MappingProxyType({"a": 1})]) == [{"a": 1}]


def test_to_plain_data_normalizes_mapping_keys_to_strings() -> None:
    assert to_plain_data({1: {"nested": {2: "x"}}}) == {"1": {"nested": {"2": "x"}}}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param({True: "yes"}, {"True": "yes"}, id="bool-key"),
        pytest.param({None: "nil"}, {"None": "nil"}, id="none-key"),
        pytest.param({("a", 1): "tuple-key"}, {"('a', 1)": "tuple-key"}, id="tuple-key"),
    ],
)
def test_to_plain_data_normalizes_various_key_types(value: object, expected: object) -> None:
    assert to_plain_data(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(
            {"items": (1, [2, {"k": (3,)}])},
            {"items": [1, [2, {"k": [3]}]]},
            id="dict-list-tuple-mix",
        ),
        pytest.param(
            MappingProxyType({1: (MappingProxyType({"a": [1, 2]}),)}),
            {"1": [{"a": [1, 2]}]},
            id="mappingproxy-nested",
        ),
    ],
)
def test_to_plain_data_handles_nested_mixed_containers(value: object, expected: object) -> None:
    assert to_plain_data(value) == expected


# =============================================================================
# as_str_object_dict
# =============================================================================


def test_as_str_object_dict_accepts_dict() -> None:
    assert as_str_object_dict({"key": "value"}, field_name="test") == {"key": "value"}


def test_as_str_object_dict_accepts_mapping_proxy() -> None:
    proxy = MappingProxyType({"a": 1})
    result = as_str_object_dict(proxy, field_name="test")
    assert result == {"a": 1}
    assert type(result) is dict


def test_as_str_object_dict_normalizes_keys_to_strings() -> None:
    assert as_str_object_dict({1: "x"}, field_name="test") == {"1": "x"}


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param({"k": "v"}, {"k": "v"}, id="dict"),
        pytest.param(MappingProxyType({1: "x"}), {"1": "x"}, id="mappingproxy"),
        pytest.param(UserDict({"a": 1}), {"a": 1}, id="userdict"),
    ],
)
def test_as_str_object_dict_accepts_mapping_variants(value: object, expected: dict[str, object]) -> None:
    assert as_str_object_dict(value, field_name="test") == expected


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("not a mapping", id="string"),
        pytest.param(None, id="none"),
        pytest.param(42, id="int"),
    ],
)
def test_as_str_object_dict_rejects_non_mapping(value: object) -> None:
    with pytest.raises(TypeError, match="test must be a mapping"):
        as_str_object_dict(value, field_name="test")


# =============================================================================
# optional_string
# =============================================================================


def test_optional_string_accepts_none() -> None:
    assert optional_string(None, field_name="f") is None


def test_optional_string_accepts_string() -> None:
    assert optional_string("hello", field_name="f") == "hello"


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("", id="empty"),
        pytest.param(" ", id="space"),
        pytest.param("\n", id="newline"),
        pytest.param("日本語", id="unicode"),
        pytest.param("0", id="digit-string"),
    ],
)
def test_optional_string_accepts_edge_strings(value: str) -> None:
    assert optional_string(value, field_name="f") == value


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(42, id="int"),
        pytest.param(True, id="bool"),
        pytest.param(3.14, id="float"),
        pytest.param([], id="list"),
    ],
)
def test_optional_string_rejects_non_string(value: object) -> None:
    with pytest.raises(TypeError, match="f must be a string or None"):
        optional_string(value, field_name="f")


# =============================================================================
# optional_int
# =============================================================================


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(None, None, id="none"),
        pytest.param(42, 42, id="positive"),
        pytest.param(0, 0, id="zero"),
        pytest.param(-1, -1, id="negative"),
        pytest.param(2**63, 2**63, id="large-int"),
    ],
)
def test_optional_int_accepts_valid_values(value: object, expected: int | None) -> None:
    assert optional_int(value, field_name="f") == expected


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(True, id="bool-true"),
        pytest.param(False, id="bool-false"),
        pytest.param(3.14, id="float"),
        pytest.param("42", id="string"),
    ],
)
def test_optional_int_rejects_invalid_values(value: object) -> None:
    with pytest.raises(TypeError, match="f must be an int or None"):
        optional_int(value, field_name="f")


# =============================================================================
# require_int
# =============================================================================


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(42, 42, id="positive"),
        pytest.param(0, 0, id="zero"),
        pytest.param(-1, -1, id="negative"),
    ],
)
def test_require_int_accepts_valid_values(value: object, expected: int) -> None:
    assert require_int(value, field_name="f") == expected


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(None, id="none"),
        pytest.param(True, id="bool-true"),
        pytest.param(False, id="bool-false"),
        pytest.param(1.0, id="float"),
    ],
)
def test_require_int_rejects_invalid_values(value: object) -> None:
    with pytest.raises(TypeError, match="f must be an int"):
        require_int(value, field_name="f")


def test_require_int_accepts_large_integer() -> None:
    assert require_int(2**80, field_name="f") == 2**80


# =============================================================================
# optional_float
# =============================================================================


def test_optional_float_accepts_none() -> None:
    assert optional_float(None, field_name="f") is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(3.14, 3.14, id="float"),
        pytest.param(0.0, 0.0, id="zero-float"),
        pytest.param(42, 42.0, id="int-coerced"),
    ],
)
def test_optional_float_accepts_valid_values(value: object, expected: float) -> None:
    assert optional_float(value, field_name="f") == expected


def test_optional_float_coerces_int_to_float() -> None:
    result = optional_float(42, field_name="f")
    assert type(result) is float


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(float("inf"), id="positive-inf"),
        pytest.param(float("-inf"), id="negative-inf"),
        pytest.param(float("nan"), id="nan"),
    ],
)
def test_optional_float_accepts_special_float_values(value: float) -> None:
    result = optional_float(value, field_name="f")
    assert result is not None
    if math.isnan(value):
        assert math.isnan(result)
    else:
        assert result == value


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(True, id="bool-true"),
        pytest.param(False, id="bool-false"),
        pytest.param("3.14", id="string"),
        pytest.param(complex(1, 2), id="complex"),
    ],
)
def test_optional_float_rejects_invalid_values(value: object) -> None:
    with pytest.raises(TypeError, match="f must be a float or None"):
        optional_float(value, field_name="f")


# =============================================================================
# string_tuple
# =============================================================================


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(None, (), id="none"),
        pytest.param([], (), id="empty-list"),
        pytest.param(["a", "b"], ("a", "b"), id="string-list"),
        pytest.param(("x", "y"), ("x", "y"), id="string-tuple"),
        pytest.param(["", " ", "日本語"], ("", " ", "日本語"), id="edge-strings"),
    ],
)
def test_string_tuple_accepts_valid_values(value: object, expected: tuple[str, ...]) -> None:
    assert string_tuple(value, field_name="f") == expected


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("not a list", id="string"),
        pytest.param(123, id="int"),
        pytest.param({"a", "b"}, id="set"),
        pytest.param((item for item in ["x"]), id="generator"),
    ],
)
def test_string_tuple_rejects_non_sequence(value: object) -> None:
    with pytest.raises(TypeError, match="f must be a sequence of strings"):
        string_tuple(value, field_name="f")


def test_string_tuple_rejects_non_string_item() -> None:
    with pytest.raises(TypeError, match=r"f\[1\] must be a string"):
        string_tuple(["ok", 42], field_name="f")


def test_string_tuple_includes_field_name_in_error() -> None:
    with pytest.raises(TypeError, match="my_field must be a sequence"):
        string_tuple(123, field_name="my_field")
