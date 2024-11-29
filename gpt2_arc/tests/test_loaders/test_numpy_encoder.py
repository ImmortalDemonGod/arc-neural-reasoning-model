from gpt2_arc.src.data.loaders import NumpyEncoder
import pytest
import torch
import numpy as np
import json

class TestNumpyEncoder:
    """Test suite for NumpyEncoder functionality."""

    def test_numpy_array_encoding(self):
        data = np.array([[1, 2], [3, 4]])
        encoded = json.dumps({'array': data}, cls=NumpyEncoder)
        decoded = json.loads(encoded)
        assert decoded['array'] == [[1, 2], [3, 4]]

    def test_torch_tensor_encoding(self):
        data = torch.tensor([[1., 2.], [3., 4.]])
        encoded = json.dumps({'tensor': data}, cls=NumpyEncoder)
        decoded = json.loads(encoded)
        assert decoded['tensor'] == [[1.0, 2.0], [3.0, 4.0]]

    def test_mixed_type_encoding(self):
        data = {
            'array': np.array([1, 2]),
            'int': np.int32(5),
            'float': np.float64(3.14),
            'list': [1, 2, 3]
        }
        encoded = json.dumps(data, cls=NumpyEncoder)
        decoded = json.loads(encoded)
        assert decoded['array'] == [1, 2]
        assert decoded['int'] == 5
        assert decoded['float'] == 3.14
        assert decoded['list'] == [1, 2, 3]
