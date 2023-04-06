import pytest

from sundry import find_intervals


class TestFindIntervals:
    def test_find_intervals(self, find_intervals_data):
        if find_intervals_data[1] is None:
            res = find_intervals(find_intervals_data[0])
        else:
            res = find_intervals(find_intervals_data[0], find_intervals_data[1])
        assert res == find_intervals_data[2]

    def test_find_intervals_invalid_parameters(self, find_intervals_invalid):
        assert find_intervals(find_intervals_invalid[0], find_intervals_invalid[1]) == find_intervals_invalid[2]

    @pytest.mark.xfail(reason="Test exception case.")
    def test_find_intervals_fail(self, find_intervals_fail):
        assert find_intervals(find_intervals_fail[0], find_intervals_fail[1]) == find_intervals_fail[2]
