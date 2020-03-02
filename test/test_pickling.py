import pickle

from torchstruct import TensorStruct


def test_tensorstruct_should_be_serializable_and_deserializable():
    x = TensorStruct.zeros({
        'a': (10, 2),
        'b': (10, 3)
    })
    x_ = pickle.loads(pickle.dumps(x))
