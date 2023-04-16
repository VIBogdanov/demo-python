from count_items import count_items


def test_data_list(data_list, data_item, operation, data_results):
    res = count_items(data_list, data_item, operation)
    assert res == data_results.get(''.join([data_item, '-', operation]), -1)


def test_data_invalid_parameters(data_list):
    assert count_items(data_list, '12') == 0
    assert count_items(data_list, 'zxc') == 0
    assert count_items(data_list, '0', 'abc') is None
    assert count_items(data_list, '27', 'abc') is None
    assert count_items(data_list, 0) == 0
    assert count_items(data_list, 14.27) == 0
