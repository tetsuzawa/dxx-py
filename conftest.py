import pytest
import os

import numpy as np

import dxx


@pytest.fixture(scope="module")
def mock_data_file() -> str:
    mock_file_name = "mock.DSB"
    sampling_freq = 48000
    mock_data = np.arange(5 * sampling_freq, dtype=np.int16)
    dxx.write(mock_file_name, mock_data)
    yield mock_file_name
    os.remove(mock_file_name)
