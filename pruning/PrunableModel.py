from typing import Optional
import torch
from pruning.PrunableEncoderLayer import PrunableEncoderLayer
from polp.nn.layers.attention import AttentionHeads, QkvMode, ScaledDotProductAttention, SelfAttention
from polp.nn.layers.feedforward import PointwiseFeedForward
from polp.nn.layers.transformer import TransformerDropouts, TransformerLayerNorms
from polp.nn.models.roberta.config import RoBERTaConfig
from polp.nn.models.roberta.encoder import RoBERTaEncoder
from torch import Tensor
from polp.nn.layers.attention import AttentionMask
from polp.nn.models.output import ModelOutput
from pruning.AttenPruner import IQRPruner, StaticPruner
from pruning.metric_logger import MetricLogger
import logging

from pruning.utils import compare_kl_divergence

logger = logging.getLogger(__name__)
metric_logger = MetricLogger()

class PrunableModel(RoBERTaEncoder):
    """
    An RoBERTa-based model that supports attention-based pruning. Identical to the building blocks of the
    RoBERTa model, it only overrides the encoding layers with ones that allow for pruning.
    """

    def __init__(
            self,
            config: RoBERTaConfig,
            *,
            device: Optional[torch.device] = None):

        super().__init__(config=config, device=device)
        alpha = float(metric_logger.get('alpha'))
        logger.info(f"PrunableModel initiated with alpha={alpha}")
        pruning_ratio = 0.65  # 固定保留35%的token
        self.last_pruning_mask = None
        self.layers = torch.nn.ModuleList(
            [
                PrunableEncoderLayer(
                    # pruner=IQRPruner(idx=_, alpha=alpha),
                    pruner=StaticPruner(idx=_, alpha=alpha, pruning_ratio=pruning_ratio),
                    attention_layer=SelfAttention(
                        attention_heads=AttentionHeads.uniform(
                            config.layer.attention.n_query_heads
                        ),
                        attention_scorer=ScaledDotProductAttention(
                            dropout_prob=config.layer.attention.dropout_prob,
                            linear_biases=None,
                            output_attention_probs=config.layer.attention.output_attention_probs,
                        ),
                        hidden_width=self.hidden_width,
                        qkv_mode=QkvMode.SEPARATE,
                        rotary_embeds=None,
                        use_bias=config.layer.attention.use_bias,
                        device=device,
                    ),
                    feed_forward_layer=PointwiseFeedForward(
                        activation=config.layer.feedforward.activation.module(),
                        hidden_width=self.hidden_width,
                        intermediate_width=config.layer.feedforward.intermediate_width,
                        use_bias=config.layer.feedforward.use_bias,
                        use_gate=config.layer.feedforward.use_gate,
                        device=device,
                    ),
                    dropouts=TransformerDropouts.layer_output_dropouts(
                        config.layer.dropout_prob
                    ),
                    layer_norms=TransformerLayerNorms(
                        attn_residual_layer_norm=self.layer_norm(),
                        ffn_residual_layer_norm=self.layer_norm(),
                    ),
                    use_parallel_attention=config.layer.attention.use_parallel_attention,
                )
                for _ in range(config.layer.n_hidden_layers)
            ]
        )

    def forward(
            self,
            piece_ids: Tensor,
            attention_mask: AttentionMask,
            *,
            positions: Optional[Tensor] = None,
            type_ids: Optional[Tensor] = None,
    ) -> ModelOutput:
        embeddings = self.embeddings(piece_ids, positions=positions, type_ids=type_ids)
        layer_output = embeddings
        layer_outputs = []
        attention_probes = []
        for layer in self.layers:
            if self.config.layer.attention.output_attention_probs:
                layer_output, layer_attention_probes, updated_mask, scores, _ = layer(layer_output, attention_mask)
                attention_mask = updated_mask
                attention_probes.append(layer_attention_probes)
            else:
                layer_output, _ = layer(layer_output, attention_mask)
            layer_outputs.append(layer_output)
        return ModelOutput(all_outputs=[embeddings, *layer_outputs], attention_probs=attention_probes)

        # prev_scores = None
        # for layer_idx, layer in enumerate(self.layers):
        #     skip_layer = False
        #     if self.config.layer.attention.output_attention_probs:
        #         if layer_idx == 0:
        #             # 第一层始终保留
        #             layer_output, layer_attention_probes, updated_mask, scores, _ = layer(layer_output, attention_mask)
        #             attention_mask = updated_mask
        #             attention_probes.append(layer_attention_probes)
        #             prev_scores = scores
        #         else:
        #             # 预计算当前层 scores（轻量级一次）
        #             _, layer_attention_probes, _, curr_scores, _ = layer(layer_output, attention_mask)
        #             # 调用封装的 KL 散度比较函数
        #             skip_layer, kl_div = compare_kl_divergence(curr_scores, prev_scores, layer_idx)
        #             if skip_layer:
        #                 continue
        #                 # logger.info(
        #                 #     f"[PrunableModel] Skipping encoder layer {layer_idx} (KL divergence {kl_div} below threshold).")
        #             else:
        #                 layer_output, layer_attention_probes, updated_mask, scores, _ = layer(layer_output,
        #                                                                                       attention_mask)
        #                 attention_mask = updated_mask
        #                 prev_scores = scores
        #             attention_probes.append(layer_attention_probes)
        #
        #     else:
        #         layer_output, _ = layer(layer_output, attention_mask)
        #     if not skip_layer:
        #         layer_outputs.append(layer_output)
        #     if layer_idx == len(self.layers) - 1 and not skip_layer:
        #         self.last_pruning_mask = attention_mask.bool_mask.squeeze(1).squeeze(1)
        #
        # return ModelOutput(all_outputs=[embeddings, *layer_outputs], attention_probs=attention_probes)

    def get_last_pruning_mask(self):
        return self.last_pruning_mask
