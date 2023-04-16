import pytest

from sundry import find_intervals, find_nearest_number


class TestFindIntervals:
    def test_find_intervals(self, find_intervals_data):
        if find_intervals_data[1] is ...:
            res = find_intervals(find_intervals_data[0])
        else:
            res = find_intervals(find_intervals_data[0], find_intervals_data[1])
        assert res == find_intervals_data[2]

    def test_find_intervals_invalid_parameters(self, find_intervals_invalid):
        if find_intervals_invalid[1] is ...:
            assert find_intervals(find_intervals_invalid[0]) == find_intervals_invalid[2]
        else:
            assert find_intervals(find_intervals_invalid[0], find_intervals_invalid[1]) == find_intervals_invalid[2]

    @pytest.mark.xfail(reason="Test exception case.")
    def test_find_intervals_fail(self, find_intervals_fail):
        assert find_intervals(find_intervals_fail[0], find_intervals_fail[1]) == find_intervals_fail[2]


class TestFindNearestNumber:
    def test_find_nearest_number_int(self, numbers_int, arguments_list, capsys):
        param_str = '\n'
        for arguments in arguments_list(numbers_int["position"]):
            res = find_nearest_number(numbers_int["number"], **arguments)
            param_str = ''.join([param_str, f'number={numbers_int["number"]}, '])
            for param_name, param_val in arguments.items():
                param_str = ''.join([param_str, f'{param_name}={param_val}, '])
            with capsys.disabled():
                print(''.join([param_str, f'result = {res}']))
            param_str = ''
            if numbers_int["result"] is None:
                assert res is numbers_int["result"]
            else:
                assert res == numbers_int["result"]

    def test_find_nearest_number_str(self, numbers_int, arguments_list, capsys):
        numbers_int["number"] = str(numbers_int["number"])
        self.test_find_nearest_number_int(numbers_int, arguments_list, capsys)

    def test_find_nearest_number_invalid_parameters(self):
        assert find_nearest_number('abc') is None
        assert find_nearest_number('812abc') is None
        assert find_nearest_number('153.12') is None
        assert find_nearest_number(153.72) == 135
        assert find_nearest_number(b'number') is None
        assert find_nearest_number(None) is None
