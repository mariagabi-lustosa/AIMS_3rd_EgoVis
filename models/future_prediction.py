# Copyright (c) Facebook, Inc. and its affiliates.

"""
Implementation of the future features prediction models.
    Input: (B, C)
    Output: (B, C)
"""
from pathlib import Path
import sys
import torch
import torch.nn as nn
import transformers
import logging

import hydra
from common.cluster import KmeansAssigner


class Identity(nn.Module):
    """Wrapper around the Identity fn to drop target_shape etc."""
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features

    def forward(self, feats, target_shape=None):
        del target_shape  # not needed here
        return feats, feats, {}, {}

    @property
    def output_dim(self):
        return self.in_features


class MLP(nn.Module):
    def __init__(self, in_features, num_layers=2):
        super().__init__()
        self.in_features = in_features
        layers = [[nn.Linear(in_features, in_features),
                   nn.ReLU(inplace=True)] for _ in range(num_layers)]
        # Flatten, remove the last ReLU, and create a sequential
        self.model = nn.Sequential(
            *([item for sublist in layers for item in sublist][:-1]))

    def forward(self, feats, target_shape=None):
        del target_shape
        return feats, self.model(feats), {}, {}

    @property
    def output_dim(self):
        return self.in_features


class AVTh(nn.Module):
    """AVT head architecture."""
    def __init__(
            self,
            in_features: int,
            output_len: int = -1,
            output_len_eval: int = -1,  # Same as output_len, used during eval
            avg_last_n: int = -1,
            inter_dim: int = 768,
            future_pred_loss: hydra.types.TargetConf = None,
            return_past_too: bool = False,
            drop_last_n: int = 0,
            quantize_before_rollout: bool = False,
            # This is only relevant when in_features=1 and input is
            # clustered, or if on the fly cluster assgn is requested
            assign_to_centroids: str = None,
            num_cluster_centers: int = 50000,
            freeze_encoder_decoder: bool = False,
            **kwargs):
        super().__init__()
        self.assign_to_centroids = assign_to_centroids
        if self.assign_to_centroids:
            # Since we will be assign the features
            assert in_features != 1
            self.assigner = KmeansAssigner(assign_to_centroids)
            assert self.assigner.num_clusters == num_cluster_centers
        if in_features == 1 or assign_to_centroids:
            self.encoder = nn.Embedding(num_cluster_centers, inter_dim)
        else:
            self.encoder = nn.Linear(in_features, inter_dim, bias=False)
        self.decoder = nn.Linear(inter_dim, in_features, bias=False)
        # If encoder is an embedding, then tie up the weights
        if isinstance(self.encoder, nn.Embedding):
            self.decoder.weight = self.encoder.weight
        if freeze_encoder_decoder:
            self.encoder.weight.requires_grad = False
            self.decoder.weight.requires_grad = False
        # This already has the LayerNorm inside residual, as Naman suggested.
        self.gpt_model = transformers.GPT2Model(
            transformers.GPT2Config(n_embd=inter_dim,
                                    vocab_size=in_features,
                                    use_cache=True,
                                    **kwargs))
        # Not needed, encoder will take care of it.
        del self.gpt_model.wte
        self.output_len = output_len
        self.output_len_eval = output_len_eval
        self.avg_last_n = avg_last_n
        self.inter_dim = inter_dim
        self.in_features = in_features
        if future_pred_loss is not None:
            self.future_pred_loss = hydra.utils.instantiate(future_pred_loss,
                                                            reduction='none')
        else:
            self.future_pred_loss = None
        self.return_past_too = return_past_too
        self.drop_last_n = drop_last_n
        # Set this, if want to quantize the prediction (using top-1) and
        # re-encode, as opposed to using the soft predicted feature
        self.quantize_before_rollout = quantize_before_rollout

    def forward(self, feats, target_shape):
        """
        Args:
            feats: tensor of shape (B, T, C)
            target_shape: shape of the output (B, T', n_output)
        """
        addl_endpoints = {}
        if feats.ndim == 2:
            # add back the temporal dimension, which was likely mean pooled
            feats = feats.unsqueeze(1)
        # Decide the output len based on the target_shape
        if len(target_shape) == 3:
            output_len = target_shape[1]
        elif self.training or self.output_len_eval < 0:
            # If training mode or output_len for eval has not been set
            output_len = self.output_len
        else:  # eval mode
            output_len = self.output_len_eval
        # Keep track
        full_inp_feats = feats
        if self.assign_to_centroids:
            # Unsqueeze only to be compatible with the 1 channel inputs -- that
            # will get squeezed out later
            feats = self.assigner(feats).unsqueeze(-1)
        # The time dimension in already in the middle -> B, T, C
        # That's what huggingface version needs:
        # (batch_size, sequence_length, hidden_size)
        if self.in_features == 1 or self.assign_to_centroids:
            # This is a quantized input, so cast it to long, and remove the
            # last singleton dimension
            assert feats.size(-1) == 1
            feats = feats.squeeze(-1).long()
        # Keep only the first N, this is used when the model is given
        # input more frames than it should be using for prediction. The other
        # future is used to incur loss during training, but shouldn't otherwise
        # be used, so dropping those features
        full_orig_feats = feats
        inp_feats = full_inp_feats
        if self.drop_last_n != 0:
            logging.warning('This should be used very carefully, ideally only '
                            'for debugging. The padding can lead to some '
                            'frames from the actual clip to leak into the '
                            'past clip, even after dropping last n. So even '
                            'after dropping the model might end up seeing '
                            'frames that are beyond the tau_a.')
            feats = feats[:, :-self.drop_last_n]
            inp_feats = inp_feats[:, :-self.drop_last_n]
        # Keep track
        orig_feats_len = feats.size(1)
        # Reduce the dimensionality, since not using the GPT encoding matrix,
        # since I don't have a "token" representation
        feats = self.encoder(feats)
        orig_feats_encoded = feats
        past = None
        all_outputs = []
        all_outputs_decoded = []
        for output_id in range(output_len):
            pred_so_far = sum([el.size(1) for el in all_outputs])
            position_ids = torch.arange(pred_so_far,
                                        pred_so_far + feats.size(1),
                                        dtype=torch.long,
                                        device=feats.device)
            # The past output will encode the previous past AND the new input
            # (you can check the output, it keeps increasing)
            # Got this from
            # https://huggingface.co/transformers/quickstart.html#using-the-past
            outputs = self.gpt_model(inputs_embeds=feats,
                                     past_key_values=past,
                                     position_ids=position_ids)
            last_hidden_state = outputs.last_hidden_state
            past = outputs.past_key_values
            all_outputs.append(last_hidden_state)
            # For visualization later, if output_attentions was passed into gpt
            if outputs.attentions is not None:
                # dimensions will be (batch_size, nlayers, nheads, seqlen, seqlen)
                addl_endpoints[f'gpt2_att_{output_id}'] = torch.stack(
                    outputs.attentions).transpose(0, 1)
            # Map back to the original feature dimension
            all_outputs_decoded.append(self.decoder(last_hidden_state))
            # hidden_states[-1] or last_hidden_state is the embedding from the
            # final layer. Not using logits (earlier was using the LMHead model
            # that returned logits) since that is already decoded to vocab size
            # and I want to have control over the weights of that final matrix
            # Also, the input for the next would be encodings, so need to
            # access the encodings directly
            if self.quantize_before_rollout:
                assert isinstance(self.encoder, nn.Embedding)
                feats = self.encoder(
                    all_outputs_decoded[-1][:, -1:, :].argmax(dim=-1))
            else:
                feats = last_hidden_state[:, -1:, :]
        all_outputs = torch.cat(all_outputs, dim=1)
        all_outputs_decoded = torch.cat(all_outputs_decoded, dim=1)
        # Compute a loss on future prediction (teacher forced)
        losses = {}
        if self.future_pred_loss is not None:
            num_elts_for_loss = min(full_orig_feats.size(1),
                                    all_outputs_decoded.size(1))
            losses = {
                'feat':
                self.future_pred_loss(
                    all_outputs_decoded[:, :num_elts_for_loss - 1],
                    full_orig_feats[:, 1:num_elts_for_loss])
            }
        # Set all_output as the final output features, and prev as the
        # structure to use to get the original features of past
        if self.in_features == 1:
            prev = orig_feats_encoded
            # all_outputs contains the hidden states, the best we will get
            # anyway, so that doesn't change
        elif self.assign_to_centroids:
            prev = inp_feats  # For this, I have the orig feats, so use that
            # For prediction, use the predicted cluster centers, but use
            # features from the original kmeans, not what the embeddings
            # that were learnt.. it didn't work with them
            all_outputs = self.assigner(all_outputs_decoded.argmax(dim=-1))
        else:
            prev = inp_feats
            all_outputs = all_outputs_decoded
        # Return the actual predictions
        if self.return_past_too:
            # Pad in the GT past (no point using the predicted past when
            # we have the actual past)
            final = torch.cat((prev, all_outputs[:, orig_feats_len - 1:, :]),
                              dim=1)
        elif output_len > 0:
            final = all_outputs[:, -output_len:]
        else:
            final = all_outputs
        if self.avg_last_n > 0:
            final = torch.mean(final[:, -self.avg_last_n:, :], dim=1)
        # compute the past feature.
        assert prev.size(1) == orig_feats_len, (
            'If not, need to figure how to deal')
        # Now keep the old feature for the first one, and return the predicted
        # features shifted by 1 for the rest -- which are as predicted by
        # GPT
        updated_past_feat = torch.cat(
            [prev[:, :1, :], all_outputs[:, :(orig_feats_len - 1)]], dim=1)
        return updated_past_feat, final, losses, addl_endpoints

    @property
    def output_dim(self):
        if self.in_features == 1:
            return self.inter_dim  # since it will return encoded features
        # else, it will decode it back to the original feat dimension
        return self.in_features


class VJEPAViTBFuturePredictor(nn.Module):
    """Official V-JEPA 2.1 ViT-B/16 predictor adapted to feature sequences.

    The official V-JEPA predictor operates on context tokens and masked target
    positions. Here we project AVT feature sequences into the ViT-B token
    space, mask the next temporal step, and use the official predictor blocks
    to produce the future embedding consumed by the action head.
    """
    def __init__(self,
                 in_features: int,
                 embed_dim: int = 768,
                 predictor_embed_dim: int = 384,
                 teacher_embed_dim: int = 1664,
                 depth: int = 12,
                 num_heads: int = 12,
                 output_len: int = 1,
                 max_len: int = 256,
                 num_mask_tokens: int = 8,
                 use_rope: bool = False,
                 use_sdpa: bool = False,
                 checkpoint_path: str = None,
                 checkpoint_url: str = (
                     'https://dl.fbaipublicfiles.com/vjepa2/'
                     'vjepa2_1_vitb_dist_vitG_384.pt'),
                 load_pretrained_predictor: bool = False,
                 future_pred_loss: hydra.types.TargetConf = None,
                 freeze_input_projection: bool = False,
                 freeze_all_but_last_n_blocks: int = 0):
        super().__init__()
        self.in_features = in_features
        self.embed_dim = embed_dim
        self.output_len = output_len
        self.max_len = max_len
        self.checkpoint_path = checkpoint_path
        self.checkpoint_url = checkpoint_url
        self.use_rope = use_rope
        self.use_sdpa = use_sdpa
        self.input_projection = nn.Linear(in_features, embed_dim, bias=False)
        vit_predictor_ctor = self._get_official_predictor_ctor()
        self.predictor = vit_predictor_ctor(
            img_size=(1, 1),
            patch_size=1,
            num_frames=max_len + output_len,
            tubelet_size=1,
            use_mask_tokens=True,
            embed_dim=embed_dim,
            predictor_embed_dim=predictor_embed_dim,
            teacher_embed_dim=teacher_embed_dim,
            depth=depth,
            num_heads=num_heads,
            num_mask_tokens=num_mask_tokens,
            use_rope=use_rope,
            uniform_power=False,
            use_sdpa=use_sdpa,
            use_silu=False,
            wide_silu=True,
            n_output_distillation=1,
            return_all_tokens=False,
            img_temporal_dim_size=1,
        )
        if not use_rope and not hasattr(self.predictor, 'predictor_pos_embed'):
            self.predictor.predictor_pos_embed = nn.Parameter(
                torch.zeros(1, self.predictor.num_patches, predictor_embed_dim))
        if not use_sdpa:
            for module in self.predictor.modules():
                if hasattr(module, 'use_sdpa'):
                    module.use_sdpa = False
        self.decoder = nn.Linear(self.predictor.predictor_proj.out_features,
                                 in_features)
        if future_pred_loss is not None:
            self.future_pred_loss = hydra.utils.instantiate(future_pred_loss,
                                                            reduction='none')
        else:
            self.future_pred_loss = None
        self._reset_parameters()
        if load_pretrained_predictor:
            self._load_official_predictor_weights()
        self._apply_freezing(freeze_input_projection,
                             freeze_all_but_last_n_blocks)

    @staticmethod
    def _add_vjepa2_to_pythonpath():
        vjepa2_root = Path(__file__).resolve().parents[1] / 'external' / 'vjepa2'
        vjepa2_root_str = str(vjepa2_root)
        if vjepa2_root.exists() and vjepa2_root_str not in sys.path:
            sys.path.insert(0, vjepa2_root_str)

    @classmethod
    def _get_official_predictor_ctor(cls):
        cls._add_vjepa2_to_pythonpath()
        from app.vjepa_2_1.models.predictor import vit_predictor
        return vit_predictor

    def _reset_parameters(self):
        nn.init.normal_(self.input_projection.weight, std=0.02)
        if self.input_projection.bias is not None:
            nn.init.constant_(self.input_projection.bias, 0)
        if hasattr(self.predictor, 'predictor_pos_embed'):
            nn.init.normal_(self.predictor.predictor_pos_embed, std=0.02)

    @staticmethod
    def _clean_backbone_key(state_dict):
        cleaned = {}
        for key, val in state_dict.items():
            key = key.replace('module.', '')
            key = key.replace('backbone.', '')
            cleaned[key] = val
        return cleaned

    def _load_official_predictor_weights(self):
        if self.checkpoint_path:
            checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        else:
            checkpoint = torch.hub.load_state_dict_from_url(
                self.checkpoint_url, map_location='cpu')
        predictor_state_dict = self._clean_backbone_key(
            checkpoint['predictor'])
        missing_keys, unexpected_keys = self.predictor.load_state_dict(
            predictor_state_dict, strict=False)
        allowed_missing = {'predictor_pos_embed'}
        allowed_unexpected = {
            'predictor_proj_context.weight',
            'predictor_proj_context.bias',
        }
        assert set(missing_keys).issubset(allowed_missing) and set(
            unexpected_keys).issubset(allowed_unexpected), (
                f'Unexpected predictor load mismatch: {missing_keys} '
                f'{unexpected_keys}')

    @staticmethod
    def _freeze_module(module):
        for param in module.parameters():
            param.requires_grad = False

    def _apply_freezing(self,
                        freeze_input_projection: bool,
                        freeze_all_but_last_n_blocks: int):
        if freeze_input_projection:
            self._freeze_module(self.input_projection)
        if freeze_all_but_last_n_blocks <= 0:
            return
        nlayers = len(self.predictor.predictor_blocks)
        if freeze_all_but_last_n_blocks >= nlayers:
            blocks_to_freeze = nlayers
        else:
            blocks_to_freeze = nlayers - freeze_all_but_last_n_blocks
        self._freeze_module(self.predictor.predictor_embed)
        for block in self.predictor.predictor_blocks[:blocks_to_freeze]:
            self._freeze_module(block)

    def _predict_future(self, feats):
        context_tokens = self.input_projection(feats)
        batch_size, num_context, _ = context_tokens.shape
        masks_x = torch.arange(num_context,
                               device=feats.device,
                               dtype=torch.long).unsqueeze(0).repeat(
                                   batch_size, 1)
        masks_y = torch.arange(num_context,
                               num_context + self.output_len,
                               device=feats.device,
                               dtype=torch.long).unsqueeze(0).repeat(
                                   batch_size, 1)
        predicted_tokens, _ = self.predictor(
            context_tokens, masks_x=masks_x, masks_y=masks_y)
        if self.output_len == 1:
            return predicted_tokens[:, 0, :]
        return predicted_tokens

    def forward(self, feats, target_shape=None):
        del target_shape
        if feats.ndim == 2:
            feats = feats.unsqueeze(1)
        assert feats.ndim == 3, f'Expected BxTxC features, got {feats.shape}'
        assert feats.size(1) + self.output_len <= self.max_len + self.output_len, (
            f'Sequence too long for max_len={self.max_len}: {feats.size(1)}')

        predicted_future_embedding = self._predict_future(feats)
        predicted_future = self.decoder(predicted_future_embedding)
        losses = {}
        if self.training and self.future_pred_loss is not None:
            min_context = max(feats.size(1) - self.output_len, 0)
            if min_context > 0:
                loss_context = feats[:, :-self.output_len, :]
                gt_future = feats[:, -self.output_len:, :]
                teacher_pred = self._predict_future(loss_context)
                if teacher_pred.ndim == 2:
                    teacher_pred = teacher_pred.unsqueeze(1)
                decoded_future = self.decoder(teacher_pred)
                losses['feat'] = self.future_pred_loss(decoded_future,
                                                       gt_future)
        return feats, predicted_future, losses, {
            'future_predicted_embedding': predicted_future_embedding,
            'future_predicted_feature': predicted_future,
        }

    @property
    def output_dim(self):
        return self.in_features
