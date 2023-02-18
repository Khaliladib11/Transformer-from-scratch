import copy
import torch.nn as nn


def replicate(block, N=6) -> nn.ModuleList:
    """
    Method to replicate the existing block to N set of blocks
    :param block: class inherited from nn.Module, mainly it is the encoder or decoder part of the architecture
    :param N: the number of stack, in the original paper they used 6
    :return: a set of N blocks
    """
    block_stack = nn.ModuleList([copy.deepcopy(block) for _ in range(N)])
    return block_stack
