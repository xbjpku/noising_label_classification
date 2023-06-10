from torch.nn import CrossEntropyLoss
from torch.nn.modules.loss import _WeightedLoss
from typing import Callable, Optional
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn

def get_alpha(step):
    alpha1 = torch.linspace(0.0, 0.0, steps=400)
    alpha2 = torch.linspace(0.0, 0.1, steps=400)
    alpha3 = torch.linspace(0.1, 1, steps=400)
    alpha4 = torch.linspace(1, 2, steps=1000)
    alpha5 = torch.linspace(2, 2.5, steps=1000)
    alpha6 = torch.linspace(2.5, 3.3, steps=2000)
    alpha7 = torch.linspace(3.3, 5, steps=2000)
     
    alpha = torch.concatenate((alpha1, alpha2, alpha3, alpha4, alpha5, alpha6, alpha7),axis=0)
    return alpha[step]

class PeerLoss(_WeightedLoss):
    
    __constants__ = ['ignore_index', 'reduction', 'label_smoothing']
    ignore_index: int
    label_smoothing: float

    def __init__(self, weight: Optional[Tensor] = None, size_average=None, ignore_index: int = -100,
                 reduce=None, reduction: str = 'mean', label_smoothing: float = 0.0, alpha = 0.0, num_classes = 200, step = 0) -> None:
        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.alpha = alpha
        self.step = step
        self._eps = 1e-5
        self._softmax = nn.Softmax(dim=-1)
        self._nllloss = nn.NLLLoss(reduction=reduction)

    def stable_entropy(self, outputs, labels):
        return self._nllloss( torch.log( self._softmax(outputs) + self._eps ), labels )

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return self.stable_entropy(input, target) \
         - get_alpha(self.step) * self.stable_entropy(input, (torch.randint(self.num_classes, target.shape, device=target.device)))
