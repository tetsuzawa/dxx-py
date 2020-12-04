import dxx
import numpy as np


class TestDtype:
    def test_from_filename(self):
        mock_filename = "mock.DSB"
        assert dxx.Dtype.DSB == dxx.Dtype.from_filename(mock_filename)

    def test_list_names(self):
        assert ["DSA", "DFA", "DDA", "DSB", "DFB", "DDB"] == dxx.Dtype.list_names()

    def test_byte_width(self):
        dtype = dxx.Dtype.DSA
        assert 2 == dtype.byte_width
        dtype = dxx.Dtype.DFA
        assert 4 == dtype.byte_width
        dtype = dxx.Dtype.DDA
        assert 8 == dtype.byte_width
        dtype = dxx.Dtype.DSB
        assert 2 == dtype.byte_width
        dtype = dxx.Dtype.DFB
        assert 4 == dtype.byte_width
        dtype = dxx.Dtype.DDB
        assert 8 == dtype.byte_width

    def test__format_specifiers(self):
        dtype = dxx.Dtype.DSA
        assert "%d" == dtype._format_specifiers
        dtype = dxx.Dtype.DFA
        assert "%e" == dtype._format_specifiers
        dtype = dxx.Dtype.DDA
        assert "%le" == dtype._format_specifiers
        dtype = dxx.Dtype.DSB
        assert "%d" == dtype._format_specifiers
        dtype = dxx.Dtype.DFB
        assert "%e" == dtype._format_specifiers
        dtype = dxx.Dtype.DDB
        assert "%le" == dtype._format_specifiers

    def test_numpy_dtype(self):
        dtype = dxx.Dtype.DSA
        assert np.int16 == dtype.numpy_dtype
        dtype = dxx.Dtype.DFA
        assert np.float32 == dtype.numpy_dtype
        dtype = dxx.Dtype.DDA
        assert np.float64 == dtype.numpy_dtype
        dtype = dxx.Dtype.DSB
        assert np.int16 == dtype.numpy_dtype
        dtype = dxx.Dtype.DFB
        assert np.float32 == dtype.numpy_dtype
        dtype = dxx.Dtype.DDB
        assert np.float64 == dtype.numpy_dtype

    def test_is_DXA(self):
        dtype = dxx.Dtype.DSA
        assert True == dtype.is_DXA
        dtype = dxx.Dtype.DFA
        assert True == dtype.is_DXA
        dtype = dxx.Dtype.DDA
        assert True == dtype.is_DXA
        dtype = dxx.Dtype.DSB
        assert False == dtype.is_DXA
        dtype = dxx.Dtype.DFB
        assert False == dtype.is_DXA
        dtype = dxx.Dtype.DDB
        assert False == dtype.is_DXA

    def test_is_DXB(self):
        dtype = dxx.Dtype.DSA
        assert False == dtype.is_DXB
        dtype = dxx.Dtype.DFA
        assert False == dtype.is_DXB
        dtype = dxx.Dtype.DDA
        assert False == dtype.is_DXB
        dtype = dxx.Dtype.DSB
        assert True == dtype.is_DXB
        dtype = dxx.Dtype.DFB
        assert True == dtype.is_DXB
        dtype = dxx.Dtype.DDB
        assert True == dtype.is_DXB

    def test___str__(self):
        dtype = dxx.Dtype.DSA
        assert "DSA" == dtype.__str__()
        dtype = dxx.Dtype.DFA
        assert "DFA" == dtype.__str__()
        dtype = dxx.Dtype.DDA
        assert "DDA" == dtype.__str__()
        dtype = dxx.Dtype.DSB
        assert "DSB" == dtype.__str__()
        dtype = dxx.Dtype.DFB
        assert "DFB" == dtype.__str__()
        dtype = dxx.Dtype.DDB
        assert "DDB" == dtype.__str__()


def test_dxx_len_file(mock_data_file):
    assert dxx.len_file(mock_data_file) == 5 * 48000
