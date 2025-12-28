import torch.nn as nn
import collections
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


def embedding_size_from_name(name):
    return int(name.strip().split('.')[-1][:-1])


def print_dim(name, tensor):
    print("%s -> %s" % (name, tensor.size()))


def plot_loss_graph(train_losses, val_losses, save_path):
    """
    Plot training and validation loss graph and save as image.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Path to save the loss graph image (e.g., 'loss_graph.png')
    """
    import os
    
    # Create figure and axis
    plt.figure(figsize=(10, 6))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot training and validation losses
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    
    # Add labels and title
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14, fontweight='bold')
    plt.legend(loc='upper right', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Loss graph saved to: {save_path}")


def load_training_metrics(metrics_file):
    """
    Load training metrics from JSON file.
    
    Args:
        metrics_file: Path to the training_metrics.jsonl file
        
    Returns:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
    """
    import json
    import os
    
    if not os.path.exists(metrics_file):
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
    
    train_losses = []
    val_losses = []
    
    with open(metrics_file, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                train_losses.append(data.get('train_loss', 0))
                val_losses.append(data.get('val_loss', 0))
    
    return train_losses, val_losses


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
