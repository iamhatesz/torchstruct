import pytest
import torch

from torchstruct import TensorStruct


def test_struct_should_raise_if_constructed_from_invalid_data():
    with pytest.raises(AssertionError):
        _ = TensorStruct({'a': (1, 2)})


def test_struct_should_allow_to_create_single_zeros_tensor():
    t = TensorStruct.zeros((2, 3), (4, 5), dtype=torch.float64, device='cpu')
    assert t.shape == (4, 5, 2, 3)
    assert t.dtype == torch.float64
    assert t.device.type == 'cpu'


def test_struct_should_allow_to_create_nested_zeros_tensors():
    t = TensorStruct.zeros({
        'a': 5,
        'b': (10,),
        'c': (3, 14),
        'd': {
            'e': 2,
            'f': (3, 1, 4),
            'g': {
                'h': {
                    'i': (8, 2)
                }
            }
        }
    }, prefix_shape=(1,))
    td = t.data()
    assert td['a'].shape == (1, 5)
    assert td['b'].shape == (1, 10)
    assert td['c'].shape == (1, 3, 14)
    assert td['d']['e'].shape == (1, 2)
    assert td['d']['f'].shape == (1, 3, 1, 4)
    assert td['d']['g']['h']['i'].shape == (1, 8, 2)


def test_struct_tensors_should_return_list_of_tensors_in_struct():
    t = TensorStruct({
        'a': torch.ones(5),
        'b': {
            'c': {
                'd': torch.ones(5) * 2
            }
        }
    })
    ts = t.tensors()
    assert len(ts) == 2
    assert any([torch.all(torch.ones(5).eq(t_)) for t_ in ts])
    assert any([torch.all(torch.ones(5).eq(t_)) * 2 for t_ in ts])
