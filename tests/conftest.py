from collections.abc import Generator
from typing import Any

import pytest

#  ------------------- test_find_nearest_number ------------------------------------
POSITION_PREV = "previous"
POSITION_NEXT = "next"


@pytest.fixture(
    name="arguments_list",
    scope="module",
)
def fixture_arguments_list():
    def _fixture_arguments_list(position=POSITION_PREV) -> list[dict]:
        if position == POSITION_PREV:
            return [
                {},
                {"previous": True},
            ]
        elif position == POSITION_NEXT:
            return [
                {"previous": False},
            ]
        else:
            return [
                {},
            ]

    return _fixture_arguments_list


numbers_list: list[dict] = [
    {"number": 907, "result_prev": 790, "result_next": 970},
    {"number": 531, "result_prev": 513, "result_next": None},
    {"number": 135, "result_prev": None, "result_next": 153},
    {"number": 2071, "result_prev": 2017, "result_next": 2107},
    {"number": 414, "result_prev": 144, "result_next": 441},
    {"number": 123456798, "result_prev": 123456789, "result_next": 123456879},
    {"number": 123456789, "result_prev": None, "result_next": 123456798},
    {"number": 1234567908, "result_prev": 1234567890, "result_next": 1234567980},
    {"number": 273145, "result_prev": 271543, "result_next": 273154},
    {"number": 276145, "result_prev": 275641, "result_next": 276154},
]


def _get_numbers_list() -> Generator[dict[str, Any], Any, None]:
    for item in numbers_list:
        res = {
            "number": (_number := item["number"]),
            "result": (_result := item["result_prev"]),
            "position": POSITION_PREV,
            "ids": f"{_result} << {_number}",
        }
        yield res

        res = res | {
            "result": (_result := item["result_next"]),
            "position": POSITION_NEXT,
            "ids": f"{res['number']} >> {_result}",
        }
        yield res

        res = res | {
            "number": (_number := (item["number"] * -1)),
            "result": (
                _result := None
                if item["result_next"] is None
                else (item["result_next"] * -1)
            ),
            "position": POSITION_PREV,
            "ids": f"{_result} << {_number}",
        }
        yield res

        res = res | {
            "result": (
                _result := None
                if item["result_prev"] is None
                else (item["result_prev"] * -1)
            ),
            "position": POSITION_NEXT,
            "ids": f"{res['number']} >> {_result}",
        }
        yield res


@pytest.fixture(
    name="numbers_int",
    scope="module",
    params=_get_numbers_list(),
    ids=lambda item: item["ids"],
)
def fixture_numbers_list_int(request) -> dict[str, Any]:
    return request.param


#  ------------------- test_count_items ------------------------------------
_data_list: list = [
    0,
    1,
    1,
    1,
    0,
    1,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    1,
    2,
    2,
    2,
    2,
    "a",
    "a",
    "a",
]


@pytest.fixture(
    name="data_list",
    scope="module",
)
def fixture_data_list() -> list:
    return _data_list


def _get_data_items() -> list[str]:
    return [str(val) for val in set(_data_list)]


@pytest.fixture(
    name="data_item",
    params=_get_data_items(),
    ids=lambda item: str(item),
)
def fixture_data_item(request) -> str:
    return request.param


_operations_list: list[str] = ["total", "min", "max", "count"]


@pytest.fixture(
    name="operation",
    params=_operations_list,
    ids=_operations_list,
)
def fixture_operations(request) -> str:
    return request.param


_data_results: dict = {
    "0-total": 9,
    "0-min": 1,
    "0-max": 5,
    "0-count": 4,
    "1-total": 6,
    "1-min": 1,
    "1-max": 3,
    "1-count": 4,
    "2-total": 4,
    "2-min": 4,
    "2-max": 4,
    "2-count": 1,
    "a-total": 3,
    "a-min": 3,
    "a-max": 3,
    "a-count": 1,
}


@pytest.fixture(
    name="data_results",
)
def fixture_data_results() -> dict:
    return _data_results


# -------------------- test_sundry ------------------------------------

# -------------------- find_intervals ------------------------------------
# ... - означает отсутствие параметра
_find_intervals_data: list[tuple] = [
    ([1, -3, 4, 5], dict(target=9), [(2, 3)], "target=9"),
    ([1, -3, 4, 5], dict(target=0), [], "target=0"),
    # ([1, -3, 4, 5], ..., [], "target=..."),
    (
        [1, -1, 4, 3, 2, 1, -3, 4, 5, -5, 5],
        dict(target=9),
        [(0, 4), (2, 4), (1, 5), (4, 8), (7, 8), (4, 10), (7, 10)],
        "target=9",
    ),
    (
        [1, -1, 4, 3, 2, 1, -3, 4, 5, -5, 5],
        dict(target=0),
        [(0, 1), (4, 6), (8, 9), (9, 10)],
        "target=0",
    ),
    # ([1, -1, 4, 3, 2, 1, -3, 4, 5, -5, 5], ..., [], "target=..."),
]


@pytest.fixture(
    name="find_intervals_data",
    params=_find_intervals_data,
    ids=lambda item: f"({item[0]}, {item[3]}) -> {item[2]}",
)
def fixture_find_intervals_data(request) -> tuple:
    return request.param


# ... - означает отсутствие параметра
_find_intervals_invalid_parameters: list[tuple] = [
    # ([], ..., [], "target=..."),
    ([1, -3, 4, 5], dict(target=1.2), [(0, 0), (1, 2)], "target=1.2"),
    ([1, -3, 4, 5], dict(target="9"), [(2, 3)], "target='9'"),
    ([1, -3, 4, 5], dict(target="1.2"), [], "target='1.2'"),
    ([1, -3, 4, 5], dict(target=None), [], "target=None"),
    ((1, -3, 4, 5), dict(target=9.0), [(2, 3)], "target=9.0"),
]


@pytest.fixture(
    name="find_intervals_invalid",
    params=_find_intervals_invalid_parameters,
    ids=lambda item: f"({item[0]}, {item[3]}) -> {item[2]}",
)
def fixture_find_intervals_invalid_parameters(request) -> tuple:
    return request.param


_find_intervals_fail: list[tuple] = [
    ([1, -3, "4", 5], dict(target=9), [], "target=9"),
    ("'1, -3, 4, 5'", dict(target=9), [], "target=9"),
    (27, dict(target=27), [], "target=27"),
    (12.5, dict(target=12), [], "target=12"),
    (None, dict(target=0), [], "target=0"),
    (
        {
            "x": 45,
        },
        dict(target=45),
        [],
        "target=45",
    ),
]


# ', '.join(str(k)+'='+str(v) for k, v in item[1].items())
@pytest.fixture(
    name="find_intervals_fail",
    params=_find_intervals_fail,
    ids=lambda item: f"({item[0]}, {item[3]}) -> {item[2]}",
)
def fixture_find_intervals_fail(request) -> tuple:
    return request.param
