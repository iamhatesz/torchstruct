from torchstruct import keys, rdefaultdict


def test_rdefaultdict_should_create_new_dictionaries_on_demand():
    d = rdefaultdict()
    d['a']['b']['c']['d'] = 5
    assert d['a']['b']['c']['d'] == 5


def test_keys_should_return_keys_of_nested_dictionaries():
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
