import pytest
import torch

from torchstruct import TensorStruct


def test_struct_should_raise_when_given_invalid_key():
    t = TensorStruct.ones({
        'a': (10, 2),
        'b': (10, 1)
    })
    with pytest.raises(KeyError):
        _ = t['c']


def test_struct_should_return_element_when_indexing_with_string():
    t = TensorStruct.ones({
        'a': (10, 2),
        'b': {
            'c': (5, 4)
        }
    })
    t_a = t['a']
    assert isinstance(t_a, torch.Tensor)
    assert t_a.shape == (10, 2)

    t_b = t['b']
    assert isinstance(t_b, TensorStruct)
    assert t_b['c'].shape == (5, 4)


def test_struct_should_return_single_elements_when_indexing_with_int():
    t = TensorStruct.ones({
        'a': (10, 2),
        'b': (10, 1)
    })
    t_1 = t[1]
    assert isinstance(t_1, TensorStruct)
    assert t_1['a'].shape == (2,)
    assert t_1['b'].shape == (1,)


def test_struct_should_return_narrowed_elements_when_indexing_with_slice():
    t = TensorStruct.ones({
        'a': (10, 2),
        'b': (10, 1)
    })
    t_05 = t[:5]
    assert isinstance(t_05, TensorStruct)
    assert t_05['a'].shape == (5, 2)
    assert t_05['b'].shape == (5, 1)


def test_struct_should_return_narrowed_elements_when_indexing_with_tensor():
    t = TensorStruct.ones({
        'a': (10, 2),
        'b': (10, 1)
    })
    t_29 = t[torch.arange(2, 9, dtype=torch.long)]
    assert isinstance(t_29, TensorStruct)
    assert t_29['a'].shape == (7, 2)
    assert t_29['b'].shape == (7, 1)


def test_struct_should_raise_when_given_index_other_than_string_int_slice_tensor():
    t = TensorStruct.ones({
        'a': (10, 2),
        'b': (10, 1)
    })
    with pytest.raises(ValueError):
        _ = t[None]


def test_struct_should_raise_if_updating_invalid_key():
    t = TensorStruct.ones({
        'a': (10, 2),
        'b': (10, 1)
    })
    with pytest.raises(KeyError):
        t['c'] = 5


def test_struct_should_update_tensor_if_tensor_given():
    t = TensorStruct.ones({
        'a': (10, 2),
        'b': (10, 1)
    })
    t['a'] = torch.zeros_like(t['a'])
    assert t['a'][0, 0] == 0


def test_struct_should_update_dict_if_valid_dict_given():
    t = TensorStruct.ones({
        'a': {
            'b': (10, 1),
            'c': (10, 2)
        }
    })
    new_data = {
        'b': torch.zeros_like(t['a']['b']),
        'c': torch.zeros_like(t['a']['c'])
    }
    t['a'] = new_data
    assert t['a']['b'][0, 0] == 0
    assert t['a']['c'][0, 0] == 0


def test_struct_should_raise_if_updating_with_invalid_dict():
    t = TensorStruct.ones({
        'a': {
            'b': (10, 1),
            'c': (10, 2)
        }
    })
    new_data = {
        'b': torch.zeros((10, 1))
    }

    with pytest.raises(ValueError):
        t['a'] = new_data


def test_struct_should_update_dict_if_valid_struct_given():
    t = TensorStruct.ones({
        'a': {
            'b': (10, 1),
            'c': (10, 2)
        }
    })
    new_data = TensorStruct({
        'b': torch.zeros_like(t['a']['b']),
        'c': torch.zeros_like(t['a']['c'])
    })
    t['a'] = new_data
    assert t['a']['b'][0, 0] == 0
    assert t['a']['c'][0, 0] == 0


def test_struct_should_raise_if_updating_with_invalid_struct():
    t = TensorStruct.ones({
        'a': {
            'b': (10, 1),
            'c': (10, 2)
        }
    })
    new_data = TensorStruct({
        'b': torch.zeros((10, 1))
    })

    with pytest.raises(ValueError):
        t['a'] = new_data


def test_struct_should_raise_if_using_unsupported_assignment():
    t = TensorStruct.ones({
        'a': {
            'b': (10, 1),
            'c': (10, 2)
        }
    })
    with pytest.raises(ValueError):
        t['a'] = 7


def test_struct_should_update_values_from_dict_at_given_indices():
    t = TensorStruct.ones({
        'a': {
            'b': (10, 1),
            'c': (10, 2)
        }
    })
    new_data = {
        'b': torch.zeros((5, 1)),
        'c': torch.zeros((5, 2))
    }
    t['a'][2:7] = new_data
    assert t['a']['b'][2, 0] == 0
    assert t['a']['c'][2, 0] == 0


def test_struct_should_update_values_from_struct_at_given_indices():
    t = TensorStruct.ones({
        'a': {
            'b': (10, 1),
            'c': (10, 2)
        }
    })
    new_data = TensorStruct({
        'b': torch.zeros((5, 1)),
        'c': torch.zeros((5, 2))
    })
    t['a'][2:7] = new_data
    assert t['a']['b'][2, 0] == 0
    assert t['a']['c'][2, 0] == 0
