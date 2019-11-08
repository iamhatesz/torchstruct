import pytest
import torch

from torchstruct import TensorStruct


def test_struct_should_forward_torch_calls_if_single_element_in_structure():
    t = TensorStruct(torch.zeros((10, 5)))
    t_expanded = t.unsqueeze(dim=1)
    assert t_expanded.shape == (10, 1, 5)


def test_struct_should_forward_torch_calls_on_each_element_in_structure():
    t = TensorStruct({
        'points': torch.zeros((10, 2)),
        'values': torch.zeros((10, 1))
    })
    t_expanded = t.unsqueeze(dim=1)
    assert t_expanded['points'].shape == (10, 1, 2)
    assert t_expanded['values'].shape == (10, 1, 1)


def test_struct_should_return_tensor_property_if_single_element_in_structure():
    t = TensorStruct(torch.zeros((10, 5)))
    assert t.shape == (10, 5)


def test_struct_should_raise_if_getting_property_on_structure_with_multiple_tensors():
    t = TensorStruct({
        'points': torch.zeros((10, 2)),
        'values': torch.zeros((10, 1))
    })
    with pytest.raises(ValueError):
        _ = t.shape
