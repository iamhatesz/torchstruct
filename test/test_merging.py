from torchstruct import TensorStruct, cat, stack


def test_cat_should_cat_nested_tensors():
    ts = [TensorStruct.ones({
        'a': (10, 2),
        'b': (10, 3)
    }) for _ in range(5)]
    ts_ = cat(ts, dim=0)
    assert ts_['a'].shape == (50, 2)
    assert ts_['b'].shape == (50, 3)


def test_stack_should_stack_nested_tensors():
    ts = [TensorStruct.ones({
        'a': (10, 2),
        'b': (10, 3)
    }) for _ in range(5)]
    ts_ = stack(ts, dim=0)
    assert ts_['a'].shape == (5, 10, 2)
    assert ts_['b'].shape == (5, 10, 3)
