from __future__ import annotations

import operator
from collections import defaultdict
from functools import reduce
from typing import Union, Dict, Tuple, Any, Callable, List, Set, Optional

import torch

TData = Union[torch.Tensor, Dict[str, 'TData']]
TShape = Tuple[int, ...]
TComplexShape = Union[int, Tuple[int, ...], Dict[str, 'TComplexShape']]
TDevice = Union[str, torch.device]


class TensorStruct:
    def __init__(self, data: TData):
        if isinstance(data, dict):
            _assert_dict(data, lambda t: isinstance(t, torch.Tensor))
        else:
            assert isinstance(data, torch.Tensor)
        self._data = data

    def data(self) -> TData:
        """
        Return internal data representation.
        """
        return self._data

    def tensors(self) -> List[torch.Tensor]:
        """
        Return list of all tensors in this structure.
        """
        if isinstance(self._data, dict):
            return tensor_values(self._data)
        return [self._data]

    def values(self) -> List[torch.Tensor]:
        """
        Alias for `tensors()`.
        """
        return self.tensors()

    def common_size(self, dim: int) -> int:
        """
        Assumes that each tensor in this structure has the same size of `dim` dimension and returns it.
        """
        return self.tensors()[0].size(dim)

    # === Representation ===
    def __repr__(self):
        return f'TensorStruct({self._data})'

    # === Initializers ===
    @staticmethod
    def build(init_fn,
              shape: TComplexShape,
              prefix_shape: TShape,
              dtype: torch.dtype,
              device: TDevice) -> Union[TensorStruct, torch.Tensor]:
        if not isinstance(shape, dict):
            return init_fn((*prefix_shape, *_assure_iterable(shape)), dtype=dtype, device=device)
        data = rdefaultdict()
        _map_dict(data, shape, lambda s: init_fn((*prefix_shape, *_assure_iterable(s)), dtype=dtype, device=device))
        return TensorStruct(data)

    @staticmethod
    def zeros(shape: TComplexShape,
              prefix_shape: TShape = (),
              dtype: torch.dtype = torch.float32,
              device: TDevice = 'cpu') -> Union[TensorStruct, torch.Tensor]:
        return TensorStruct.build(torch.zeros, shape, prefix_shape, dtype, device)

    @staticmethod
    def ones(shape: TComplexShape,
             prefix_shape: TShape = (),
             dtype: torch.dtype = torch.float32,
             device: TDevice = 'cpu') -> Union[TensorStruct, torch.Tensor]:
        return TensorStruct.build(torch.ones, shape, prefix_shape, dtype, device)

    @staticmethod
    def empty(shape: TComplexShape,
              prefix_shape: TShape = (),
              dtype: torch.dtype = torch.float32,
              device: TDevice = 'cpu') -> Union[TensorStruct, torch.Tensor]:
        return TensorStruct.build(torch.empty, shape, prefix_shape, dtype, device)

    @staticmethod
    def randn(shape: TComplexShape,
              prefix_shape: TShape = (),
              dtype: torch.dtype = torch.float32,
              device: TDevice = 'cpu') -> Union[TensorStruct, torch.Tensor]:
        return TensorStruct.build(torch.randn, shape, prefix_shape, dtype, device)

    # === Indexing ===
    def __contains__(self, item: str) -> bool:
        if not isinstance(self._data, dict):
            return False
        return item in self._data

    def __getitem__(self, item: Union[str, int, slice, torch.Tensor]) -> Union[torch.Tensor, TensorStruct]:
        if isinstance(item, str):
            if item not in self:
                raise KeyError(f'Key not found (`{item}` given)')
            if isinstance(self._data[item], dict):
                return TensorStruct(self._data[item])
            return self._data[item]
        elif isinstance(item, int) or isinstance(item, slice) or isinstance(item, torch.Tensor):
            return TensorStruct(self._index(item))
        else:
            raise ValueError(f'Only indexing with `str`, `int`, `slice` or `torch.Tensor` is supported '
                             f'(`{type(item)}` given)')

    def _index(self, item: Union[int, slice, torch.Tensor]) -> TData:
        if isinstance(self._data, torch.Tensor):
            return self._data[item]
        d = rdefaultdict()
        _map_dict(d, self._data, lambda t: t[item])
        return d

    # === Updating ===
    def __setitem__(self, key: Union[str, int, slice, torch.Tensor], value):
        if isinstance(key, str):
            if key not in self:
                raise KeyError(f'Key not found (`{key}` given)')
            if isinstance(self._data[key], torch.Tensor) and isinstance(value, torch.Tensor):
                self._data[key] = value
            elif isinstance(self._data[key], dict) and isinstance(value, dict):
                if keys(self._data[key]) != keys(value):
                    raise ValueError('Trying to assign `dict` that does not match structure')
                self._data[key] = value
            elif isinstance(self._data[key], dict) and isinstance(value, TensorStruct):
                if keys(self._data[key]) != keys(value._data):
                    raise ValueError('Trying to assign `TensorStruct` that does not match structure')
                self._data[key] = value._data
            else:
                raise ValueError('Unsupported assignment operation')
        elif isinstance(key, int) or isinstance(key, slice) or isinstance(key, torch.Tensor):
            if isinstance(value, dict):
                if keys(self._data) != keys(value):
                    raise ValueError('Trying to assign `dict` that does not match structure')
                _update_dict_at(self._data, value, key)
            elif isinstance(value, TensorStruct):
                if keys(self._data) != keys(value._data):
                    raise ValueError('Trying to assign `TensorStruct` that does not match structure')
                _update_dict_at(self._data, value._data, key)
            else:
                raise ValueError('Unsupported assignment operation')

    # === Processing data ===
    def apply(self, fn: Callable[[torch.Tensor], torch.Tensor],
              keep_struct: bool = False) -> Union[torch.Tensor, TensorStruct]:
        if isinstance(self._data, torch.Tensor):
            return fn(self._data) if not keep_struct else TensorStruct(fn(self._data))
        d = rdefaultdict()
        _map_dict(d, self._data, fn)
        return TensorStruct(d)

    # === Forwarding PyTorch calls ===
    def __getattr__(self, item: str):
        if not hasattr(torch.Tensor, item):
            return super().__getattribute__(item)
        prop = getattr(torch.Tensor, item)
        if callable(prop):
            return lambda *args, **kwargs: self._apply_pytorch_method(prop, *args, **kwargs)
        elif isinstance(self._data, torch.Tensor):
            return getattr(self._data, item)
        else:
            raise ValueError('Property can be retrieved only from single tensor structures')

    def _apply_pytorch_method(self, method, *args, **kwargs):
        keep_struct = kwargs.get('keep_struct', False)
        # Remove `keep_struct` from `kwargs` to not be passed to PyTorch method
        if 'keep_struct' in kwargs:
            del kwargs['keep_struct']
        return self.apply(lambda t: method(t, *args, **kwargs), keep_struct=keep_struct)

    # === Pickling support ===
    def __getstate__(self):
        return {'_data': self._data}

    def __setstate__(self, state):
        self._data = state['_data']


def _assure_iterable(x):
    if isinstance(x, tuple) or isinstance(x, list):
        return x
    return x,


def _map_dict(d_out: Dict[str, Any], d_in: Dict[str, Any], fn: Callable[[Any], Any]):
    for key, value in d_in.items():
        if isinstance(value, dict):
            _map_dict(d_out[key], value, fn)
        else:
            d_out[key] = fn(value)


def _assert_dict(d: Dict[str, Any], fn: Callable[[Any], bool]):
    for key, value in d.items():
        if isinstance(value, dict):
            _assert_dict(d[key], fn)
        else:
            assert fn(value)


def _update_dict_at(base: TData, data: TData, selector: Union[int, slice, torch.Tensor]):
    for key, value in base.items():
        if isinstance(value, dict):
            _update_dict_at(base[key], data[key], selector)
        else:
            base[key][selector] = data[key]


def rdefaultdict():
    return defaultdict(rdefaultdict)


def keys(d: Dict[str, Any], prefix: Optional[Tuple[str, ...]] = None) -> Set[Tuple[str, ...]]:
    if prefix is None:
        prefix = tuple()
    k = []
    for key, value in d.items():
        if isinstance(value, dict):
            k.extend(keys(value, prefix=prefix + (key,)))
        k.append(prefix + (key,))
    return set(k)


def tensor_values(d: Dict[str, Any]) -> List[Any]:
    v = []
    for key, value in d.items():
        if isinstance(value, dict):
            v.extend(tensor_values(d[key]))
        elif isinstance(value, torch.Tensor):
            v.append(value)
    return v


def _dict_nested_get(d, keys):
    return reduce(operator.getitem, keys, d)


def _dict_nested_set(d, keys, value):
    _dict_nested_get(d, keys[:-1])[keys[-1]] = value


def cat(structs: List[TensorStruct], dim: int = 0) -> TensorStruct:
    """
    Concatenate list of `structs` along existing `dim`.
    """
    if len(structs) == 0:
        raise ValueError('At least one `TensorStruct` is required')
    s = structs[0]
    ks = keys(s.data())
    d = rdefaultdict()
    for key in ks:
        ts = tuple(_dict_nested_get(s.data(), key) for s in structs)
        _dict_nested_set(d, key, torch.cat(ts, dim=dim))
    return TensorStruct(d)


def stack(structs: List[TensorStruct], dim: int = 0) -> TensorStruct:
    """
    Stack list of `structs` along new `dim`.
    """
    if len(structs) == 0:
        raise ValueError('At least one `TensorStruct` is required')
    s = structs[0]
    ks = keys(s.data())
    d = rdefaultdict()
    for key in ks:
        ts = tuple(_dict_nested_get(s.data(), key) for s in structs)
        _dict_nested_set(d, key, torch.stack(ts, dim=dim))
    return TensorStruct(d)
