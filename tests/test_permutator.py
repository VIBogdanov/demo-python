from permutator import find_nearest_number


def test_numbers_int(numbers_int, arguments_list, capsys):
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


def test_numbers_str(numbers_int, arguments_list, capsys):
    numbers_int["number"] = str(numbers_int["number"])
    test_numbers_int(numbers_int, arguments_list, capsys)


def test_numbers_fail():
    assert find_nearest_number('abc') is None
    assert find_nearest_number('812abc') is None
    assert find_nearest_number('153.12') is None
    assert find_nearest_number(153.72) == 135
    assert find_nearest_number(b'number') is None
    assert find_nearest_number(None) is None
