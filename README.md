# torchstruct

Wrap your multiple `torch.Tensor`s into single `TensorStruct`
and use it like you are using `torch.Tensor`.

## Installation

```bash
pip install torchstruct
```

## Testing

```bash
PYTHONPATH=. pytest
```

## Examples

```python
import torch
from torchstruct import TensorStruct

# Initialization
ts = TensorStruct.zeros({
    'obs': (2,),
    'rew': (1,),
    'done': (1,)
}, prefix_shape=(10,), dtype=torch.float32, device='cpu')

raw_data = {
    'obs': torch.randn((10, 2)),
    'rew': torch.randn((10, 1)),
    'done': torch.randn((10, 1))
}

# Assigning
ts[:] = raw_data

# Indexing
ts[2:4]
ts['rew']

# Calling PyTorch methods
ts.unsqueeze(dim=0)
ts.sum(dim=0)
```
