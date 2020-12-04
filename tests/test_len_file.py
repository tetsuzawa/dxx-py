import dxx


def test_dxx_len_file(mock_data_file):
    assert dxx.len_file(mock_data_file) == 5 * 48000
