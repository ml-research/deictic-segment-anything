import torch

from .torch_utils import softor


class SoftLogic(object):
    """An class of the soft-implementation of logic operations, i.e., logical-or and logical-and.
    """

    def __init__(self):
        pass

    def _or(self, x):
        return softor(x, dim=0, gamma=0.01)

    def _and(self, x):
        return torch.prod(x, dim=0)


class LukasiewiczSoftLogic(SoftLogic):
    pass


class LNNSoftLogic(SoftLogic):
    pass


class aILPSoftLogic(SoftLogic):
    pass
