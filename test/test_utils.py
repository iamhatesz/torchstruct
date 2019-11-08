import torch

from torchstruct import keys, rdefaultdict, tensor_values


def test_rdefaultdict_should_create_new_dictionaries_on_demand():
    d = rdefaultdict()
    d['a']['b']['c']['d'] = 5
    assert d['a']['b']['c']['d'] == 5


def test_keys_should_return_keys_of_nested_dicts():
    d = {
        'a': {
            'b': True,
            'c': False
        },
        'x': {
            'y': {
                'z': {
                    'a': []
                }
            }
        }
    }
    k = keys(d)
    assert len(k) == 7
    assert 'a' in k
    assert 'a.b' in k
    assert 'a.c' in k
    assert 'x' in k
    assert 'x.y' in k
    assert 'x.y.z' in k
    assert 'x.y.z.a' in k


def test_tensor_values_should_return_values_of_nested_dicts():
    d = {
        'a': torch.ones(5),
        'b': {
            'c': torch.ones(5) * 2,
            'd': {
                'e': torch.ones(5) * 3
            }
        }
    }
    v = tensor_values(d)
    assert len(v) == 3
    assert any([torch.all(torch.ones(5).eq(t)) for t in v])
    assert any([torch.all(torch.ones(5).eq(t)) * 2 for t in v])
    assert any([torch.all(torch.ones(5).eq(t)) * 3 for t in v])
