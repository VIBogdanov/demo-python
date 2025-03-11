import pytest

from demo import find_intervals, find_nearest_number


class TestFindIntervals:
    def test_find_intervals(self, find_intervals_data):
        if find_intervals_data[1] is ...:
            res = find_intervals(find_intervals_data[0])
        else:
            res = find_intervals(find_intervals_data[0], **find_intervals_data[1])
        assert res == find_intervals_data[2]

    def test_find_intervals_invalid_parameters(self, find_intervals_invalid):
        if find_intervals_invalid[1] is ...:
            assert (
                find_intervals(find_intervals_invalid[0]) == find_intervals_invalid[2]
            )
        else:
            assert (
                find_intervals(find_intervals_invalid[0], **find_intervals_invalid[1])
                == find_intervals_invalid[2]
            )

    @pytest.mark.xfail(reason="Test exception case.")
    def test_find_intervals_fail(self, find_intervals_fail):
        assert (
            find_intervals(find_intervals_fail[0], **find_intervals_fail[1])
            == find_intervals_fail[2]
        )


class TestFindNearestNumber:
    def test_find_nearest_number_int(self, numbers_int, arguments_list, capsys):
        first_line = "\n"
        _number = numbers_int["number"]
        for arguments in arguments_list(numbers_int["position"]):
            res = find_nearest_number(_number, **arguments)
            with capsys.disabled():
                print(
                    "".join(
                        (
                            f"{first_line}find_nearest_number({_number}",
                            *(
                                f", {param_name}={param_val}"
                                for param_name, param_val in arguments.items()
                            ),
                            f") -> {res}",
                        )
                    )
                )
                first_line = ""

            if (_res := numbers_int["result"]) is None:
                assert res is _res
            else:
                assert res == _res

    def test_find_nearest_number_str(self, numbers_int, arguments_list, capsys):
        numbers_int["number"] = str(numbers_int["number"])
        self.test_find_nearest_number_int(numbers_int, arguments_list, capsys)

    @pytest.mark.parametrize(
        "args",
        [
            ("abc", None),
            ("812abc", None),
            ("153.12", None),
            (153.12, 135),
            (b"number", None),
            (None, None),
        ],
        ids=lambda item: f"find_nearest_number({repr(item[0])}) -> {item[1]}",
    )
    def test_find_nearest_number_invalid_parameters(self, args):
        arg, res = args
        if res is None:
            assert find_nearest_number(arg) is res
        else:
            assert find_nearest_number(arg) == res
