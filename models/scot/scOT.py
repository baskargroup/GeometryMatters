import torch
import torch.nn as nn
from models.base import BaseLightningModule
from scOT.model import ScOT, ScOTConfig




class scOT(BaseLightningModule):
    def __init__(self, in_channels, out_channels, depths, embed_dim, pretrained_path=None, loss=nn.MSELoss(), lr=1e-3, plot_path='./plots/', log_file='fno_log.txt'):
        super(scOT, self).__init__(lr=lr, plot_path=plot_path, log_file=log_file)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.loss = loss
        self.depths = depths
        self.embed_dim = embed_dim
        self.model_config = ScOTConfig(
                                    image_size=512,
                                    patch_size=4,
                                    num_channels=in_channels,
                                    num_out_channels=out_channels,
                                    embed_dim=self.embed_dim,
                                    depths=self.depths,
                                    num_heads=[3, 6, 12, 24],
                                    skip_connections=[2, 2, 2, 0],
                                    window_size=16,
                                    mlp_ratio=4.0,
                                    qkv_bias=True,
                                    hidden_dropout_prob=0.0,  # default
                                    attention_probs_dropout_prob=0.0,  # default
                                    drop_path_rate=0.0,
                                    hidden_act="gelu",
                                    use_absolute_embeddings=False,
                                    initializer_range=0.02,
                                    layer_norm_eps=1e-5,
                                    p=1,
                                    channel_slice_list_normalized_loss=None,
                                    residual_model="convnext",
                                    use_conditioning=None,
                                    learn_residual=False,
                                )
        
        if pretrained_path is not None:
            self.model = ScOT.from_pretrained(pretrained_path, config=self.model_config, ignore_mismatched_sizes=True)
        else:
            self.model = ScOT(config=self.model_config)

    def forward(self, x):
        return self.model(x)["output"]
