import torch.nn as nn
import collections


def embedding_size_from_name(name):
    return int(name.strip().split('.')[-1][:-1])


def print_dim(name, tensor):
    print("%s -> %s" % (name, tensor.size()))


class RNNWrapper(nn.Module):
    """
    Wrapper around GRU or LSTM RNN. If underlying RNN is GRU, this wrapper does nothing, it just forwards inputs and
    outputs. If underlying RNN is LSTM this wrapper ignores LSTM cell state (s) and returns just hidden state (h).
    This wrapper allows us to unify interface for GRU and LSTM so we don't have to treat them differently.
    """

    LSTM = 'LSTM'
    GRU = 'GRU'
    def __init__(self, rnn):
            super(RNNWrapper, self).__init__()
            assert isinstance(rnn, nn.LSTM) or isinstance(rnn, nn.GRU)
            self.rnn_type = self.LSTM if isinstance(rnn, nn.LSTM) else self.GRU
            self.rnn = rnn
    
        def forward(self, *input):
            rnn_out, hidden = self.rnn(*input)
            if self.rnn_type == self.LSTM:
                hidden, s = hidden  # ignore LSTM cell state s
            return rnn_out, hidden
    
# Metadata used to describe dataset
Metadata = collections.namedtuple('Metadata', 'vocab_size padding_idx vectors')

# -------------------------------------------------------------------------
# [New] Ground Truth Initialization
# 논문 Section 4.1: "parameters are uniformly initialized in [-0.1, 0.1]"
# -------------------------------------------------------------------------
def init_weights(model):
    """
    모든 서브 모듈의 파라미터를 [-0.1, 0.1] 범위의 Uniform Distribution으로 초기화합니다.
    Args:
        model: nn.Module
    """
    for name, param in model.named_parameters():
        # 바이어스와 웨이트 모두 uniform 초기화
        nn.init.uniform_(param.data, -0.1, 0.1)
