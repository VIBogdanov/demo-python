import pytest

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
                {"multiproc": False},
                {"multiproc": True},
            ]
        elif position == POSITION_NEXT:
            return [
                {"previous": False},
                {"previous": False, "multiproc": True},
            ]
        else:
            return [{},]
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


def _get_numbers_list() -> list[dict]:
    return (
        [
            {
                **item,
                **{
                    "result": item["result_prev"],
                    "position": POSITION_PREV,
                    "ids": str(item["result_prev"]) + " << " + str(item["number"]),
                },
            }
            for item in numbers_list
        ]
        + [
            {
                **item,
                **{
                    "number": (item["number"] * -1),
                    "result": None if item["result_next"] is None else (item["result_next"] * -1),
                    "position": POSITION_PREV,
                    "ids": "-" + str(item["result_next"]) + " << -" + str(item["number"]),
                },
            }
            for item in numbers_list
        ]
        + [
            {
                **item,
                **{
                    "result": item["result_next"],
                    "position": POSITION_NEXT,
                    "ids": str(item["number"]) + " >> " + str(item["result_next"]),
                },
            }
            for item in numbers_list
        ]
        + [
            {
                **item,
                **{
                    "number": (item["number"] * -1),
                    "result": None if item["result_prev"] is None else (item["result_prev"] * -1),
                    "position": POSITION_NEXT,
                    "ids": "-" + str(item["number"]) + " >> -" + str(item["result_prev"]),
                },
            }
            for item in numbers_list
        ]
    )


@pytest.fixture(
    name="numbers_int",
    scope="module",
    params=_get_numbers_list(),
    ids=lambda item: str(item["ids"]),
)
def fixture_numbers_list_int(request) -> dict:
    return request.param
