import torch
from torch import nn
from models import fusion_blocks

class LGFusion(nn.Module):
    def __init__(
            self,
            fusion_arch='factorized_mmi',
            fusion_layers='all',
            num_fusion_tkns=(4, 8, 4),
            fusion_mlp_ratio=1.0, fusion_attn_ratio=0.25, fusion_num_heads=12,
            drop_path=0., attn_drop=0., drop=0.,
    ):
        super(LGFusion, self).__init__()

        self.embed_dim = 768  # Assuming the feature dimension is 768
        self.fusion_arch = fusion_arch

        # Multi-modal fusion blocks and tokens
        self.num_fusion = num_fusion_tkns
        self.fusion_tokens = nn.Parameter(torch.zeros(1, sum(num_fusion_tkns), self.embed_dim))

        FusionBlock = None
        if fusion_arch == 'token':
            FusionBlock = fusion_blocks.FusionBlock_LocalAVTokens
        elif fusion_arch == 'dense_mmi':
            FusionBlock = fusion_blocks.FusionBlock_DenseAVInteractions
        elif fusion_arch == 'factorized_mmi':
            from functools import partial
            FusionBlock = partial(fusion_blocks.FusionBlock_FactorizedAVInteractions, fusion_tkns=num_fusion_tkns)

        max_depth = 12  # Assuming the maximum depth of the fusion blocks
        if fusion_layers == 'all':
            fusion_layers = set(range(max_depth))
        elif fusion_layers == 'none':
            fusion_layers = set([])
        elif isinstance(fusion_layers, int):
            fusion_layers = {fusion_layers}
        else:
            fusion_layers = set([int(l) for l in fusion_layers.split('-')])
        self.fusion_blocks = nn.ModuleList([
            None if i not in fusion_layers or FusionBlock is None else FusionBlock(
                dim=self.embed_dim, num_heads=fusion_num_heads, attn_ratio=fusion_attn_ratio, mlp_ratio=fusion_mlp_ratio, qkv_bias=True,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path, norm_layer=nn.LayerNorm)
            for i in range(max_depth)])
        self.fusion_norm = nn.LayerNorm(self.embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.fusion_tokens, std=.02)
        self.fusion_blocks.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, lang_features, graph_features, return_embs=False):
        B = lang_features.shape[0]

        # Apply blocks
        embs = []
        x_fusion = self.fusion_tokens.expand(B, -1, -1)
        nL, nG, nF = lang_features.shape[1], graph_features.shape[1], self.fusion_tokens.shape[1]
        for blk_lang, blk_graph, blk_fusion in zip(self.fusion_blocks, self.fusion_blocks, self.fusion_blocks):
            if blk_fusion is None:
                lang_features = blk_lang(lang_features)
                graph_features = blk_graph(graph_features)
            else:
                # _, _lang_features = blk_lang(torch.cat((x_fusion, lang_features), dim=1)).split((nF, nL), dim=1)
                # _, _graph_features = blk_graph(torch.cat((x_fusion, graph_features), dim=1)).split((nF, nG), dim=1)
                x_fusion = blk_fusion(x_fusion, lang_features, graph_features)
                # lang_features, graph_features = _lang_features, _graph_features
            if return_embs:
                embs.append((lang_features, graph_features, x_fusion))

        lang_features = self.fusion_norm(lang_features)
        graph_features = self.fusion_norm(graph_features)
        x_fusion = self.fusion_norm(x_fusion)

        if not return_embs:
            return lang_features, graph_features, x_fusion
        else:
            return lang_features, graph_features, x_fusion, embs