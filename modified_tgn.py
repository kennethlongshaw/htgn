from torch_geometric.nn.models import TGNMemory
from message_encoder import AttentionMessageMemory


class AttentionTGN(TGNMemory):
    """Overrides the memory module of TGN to use a cross attention-based memory system"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.gru = AttentionMessageMemory(input_channels=kwargs['message_module'].out_channels,
                                          emb_dim=kwargs['memory_dim'])
