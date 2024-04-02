""" version: cascade + value vector without position encoding """
from typing import Tuple, Optional, List, Union
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as nn_init

from transformers.models.roberta.modeling_roberta import (
    RobertaPreTrainedModel, RobertaEmbeddings, RobertaEncoder, RobertaPooler,
    BaseModelOutputWithPoolingAndCrossAttentions,
    RobertaConfig
)
from transformers.activations import gelu

MASK_TOKEN_ID = 50264 # RoBerta Tokenizer [MASK] ID

class TPBertaHead(nn.Module):
    """Single TPBertaHead for regression, binclass, multiclass"""

    def __init__(self, config, num_labels):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class TPBertaMultiTaskHead(nn.Module):
    """A TPBertaMTLHead (multi-task learning head) with multiple task-specific heads"""
    def __init__(self, config, num_classes: List[int]) -> None:
        super().__init__()
        # consists of task-specific heads for multi-task learning
        self.heads = nn.ModuleList([
            TPBertaHead(config, num_labels)
        for num_labels in num_classes])
    
    def __getitem__(self, index):
        return self.heads[index]
    
    def forward(self, i, x):
        return self[i](x)


class TPBertaEmbeddings(RobertaEmbeddings):
    """
    TPBertaEmbeddings decouple representations of feature words & numerical values
    Reference: RobertaEmbeddings
    """
    
    def forward(self, input_ids=None, input_scales=None, token_type_ids=None, position_ids=None):
        input_shape = input_ids.size()
        
        seq_length = input_shape[1]

        assert token_type_ids is not None
        assert position_ids is not None
        
        inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        v_embeddings = embeddings.clone() # embeddings for value vectors have no position info
        v_embeddings = self.LayerNorm(v_embeddings)
        v_embeddings = self.dropout(v_embeddings) 
        v_embeddings *= input_scales[..., None] # numerical vectors x their value scales
        if self.position_embedding_type == 'absolute':
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings # embeddings for QK have position info
        qk_embeddings = self.LayerNorm(embeddings)
        qk_embeddings = self.dropout(qk_embeddings)
        qk_embeddings *= input_scales[..., None] # numerical vectors x their value scales
        return qk_embeddings, v_embeddings


class IntraFeatureAttention(nn.Module):
    """Fuse each feature chunk into a single vector"""
    def __init__(self, config):
        super().__init__()
        self.W_q = nn.Linear(config.hidden_size, config.hidden_size)
        self.W_k = nn.Linear(config.hidden_size, config.hidden_size)
        self.W_v = nn.Linear(config.hidden_size, config.hidden_size)
        self.W_out = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.n_heads = config.num_attention_heads

        for m in [self.W_q, self.W_k, self.W_v, self.W_out]:
            nn_init.zeros_(m.bias)
    
    def forward(
        self,
        x: Tuple[torch.Tensor],
        query_mask: torch.Tensor, # [CLS] one-hot indicator of each feature
        input_ids: torch.Tensor,
        pad_token_id: int = 1, # roberta <pad> id
    ):
        x_qk, x_v = x
        b, l, d = x_qk.shape
        f = query_mask.sum().cpu().item() # f
        assert (b, l) == input_ids.shape
        assert (b, l) == query_mask.shape
        assert f % b == 0
        device = x_qk.device
        
        attn_thresh = torch.cat(
            [torch.where(query_mask.view(-1) == 1)[0],
            torch.tensor([b * l]).to(device)]
        ) # f + 1
        attn_left = attn_thresh[:-1] # f
        attn_right = attn_thresh[1:] 
        feature_ids = torch.arange(b * l, device=device).repeat(len(attn_left), 1) # f, b*l
        attention_mask = (feature_ids >= attn_left.view(-1, 1)) & (feature_ids < attn_right.view(-1, 1)) # f, b*l
        attention_mask = attention_mask & (input_ids.view(1, -1) != pad_token_id)

        x_qk = x_qk.reshape(-1, d) # b*l, d
        x_v = x_v.reshape(-1, d) # b*l, d
        query = x_qk[query_mask.view(-1) == 1] # f, d; [CLS] token of each feature
        q = self.W_q(query) # f, d
        k, v = self.W_k(x_qk), self.W_v(x_v) # b*l, d

        for _x in [q, k, v]:
            assert _x.shape[-1] % self.n_heads == 0
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads

        q = q.reshape(f, self.n_heads, d_head_key).transpose(0,1) # n_heads, f, _d
        k = k.reshape(b*l, self.n_heads, d_head_key).transpose(0,1) # n_heads, b*l, _d
        attention_mask = (1.0 - attention_mask.float()) * -10000
        attention = F.softmax(
            q @ k.transpose(1,2) / math.sqrt(d_head_key)
            + attention_mask[None], 
            dim=-1
        ) # n_heads, f, b*l
        attention = self.dropout(attention)
        v = v.reshape(b*l, self.n_heads, d_head_value).transpose(0,1) # n_heads, b*l, _d
        x = attention @ v # n_heads, f, _d
        x = x.transpose(0,1).reshape(f, d_head_value * self.n_heads) # f, d
        x = self.W_out(x)

        return x.reshape(b, f // b, d)


def append_prompt(x: torch.Tensor, prompt: torch.Tensor):
    b, l, d = x.shape
    assert prompt.shape[-1] == d
    if prompt.shape[0] == 1:
        prompt = prompt.repeat(b, 1, 1)
    try:
        new_prompt = torch.cat([x, prompt], dim=1)
    except Exception:
        print('x: ', x.shape)
        print('prompt: ', prompt.shape)
        raise RuntimeError('Check your prompt repeat time is equal to input batch size')
    return new_prompt


class TPBerta(RobertaPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        # TPBertaEmbeddings for numerical sensitivity consideration
        self.embeddings = TPBertaEmbeddings(config)
        # Intra-feature Attention (IFA)
        self.intra_attention = IntraFeatureAttention(config)
        # normal Roberta Encoder & Pooler
        self.encoder = RobertaEncoder(config)

        self.pooler = RobertaPooler(config) if add_pooling_layer else None

        self.post_init()
    
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings
    
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
    
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_scales: Optional[torch.Tensor] = None,
        feature_cls_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        tail_prompt: Optional[torch.Tensor] = None, # for prompt-based tuning (not used in the paper)
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        assert input_ids is not None
        
        device = input_ids.device

        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            input_scales=input_scales,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        feature_chunk = self.intra_attention(
            embedding_output,
            query_mask=feature_cls_mask,
            input_ids=input_ids,
        )
        # prompt-based tuning (not used in the paper)
        if tail_prompt is not None:
            if tail_prompt.ndim == 1:
                tail_prompt = tail_prompt.unsqueeze(0)
            l = tail_prompt.shape[1]
            tail_prompt = self.embeddings(
                input_ids=tail_prompt,
                input_scales=torch.ones_like(tail_prompt, device=input_ids.device, dtype=torch.float32),
                token_type_ids=torch.zeros_like(tail_prompt, device=input_ids.device, dtype=torch.int64),
                position_ids=torch.arange(l, device=input_ids.device).unsqueeze(0).expand_as(tail_prompt),
            )[0] # add position in prompt embeddings
            feature_chunk = append_prompt(feature_chunk, tail_prompt) # append prompt

        input_shape = feature_chunk.shape[:2]
        batch_size, feature_length = input_shape

        assert feature_chunk.shape[0] == batch_size
        attention_mask = torch.ones(((batch_size, feature_length)), device=device)

        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        encoder_outputs = self.encoder(
            feature_chunk,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class TPBertaLMHead(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x


def mask_inputs(
    input_ids: torch.Tensor, 
    token_type_ids: torch.Tensor, 
    p: float = 0.2
):
    b, l = input_ids.shape
    numerical_token_mask = token_type_ids == 1
    numerical_tokens = numerical_token_mask.sum().cpu().item()
    assert numerical_tokens % b == 0
    avg_numerical_tokens = numerical_tokens // b
    masked_loc = torch.rand(input_ids.size(), device=input_ids.device) < p
    masked_loc = masked_loc & numerical_token_mask # only mask numerical values
    if masked_loc.sum().cpu().item() == 0: # no numerical features selected
        return input_ids, None, None, 0
    _masked_loc = masked_loc[numerical_token_mask] # n_mask, (for loss mask)
    masked_input_ids = input_ids.clone()
    masked_tokens = masked_input_ids[masked_loc].clone()
    masked_input_ids[masked_loc] = MASK_TOKEN_ID
    return masked_input_ids, masked_tokens, _masked_loc, avg_numerical_tokens


class TPBertaForMTLPretrain(RobertaPreTrainedModel):
    """TP-BERTa for multi-task learning pre-training"""

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]
    base_model_prefix = 'tpberta'

    class TPBertaRankHead(nn.Module):
        """A ranker head for triplet loss"""
        def __init__(self, config) -> None:
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        def forward(self, x):
            x = self.dense(x)
            x = gelu(x)
            x = self.layer_norm(x)
            return x

    def __init__(self, config, num_classes: List[int], add_mlm_head=False, prompt_based=False):
        super().__init__(config)

        self.tpberta = TPBerta(config, add_pooling_layer=False)
        if not prompt_based:
            self.heads = TPBertaMultiTaskHead(config, num_classes)
        self.ranker = self.TPBertaRankHead(config)
        if add_mlm_head or prompt_based:
            self.mlm_head = TPBertaLMHead(config)
        self.mlm = add_mlm_head
    
        self.post_init()
    
    def resize_mlm_head(self):
        if not hasattr(self, 'mlm_head'):
            return
        if self.config.vocab_size != self.mlm_head.bias.shape[0]:
            self.mlm_head = TPBertaLMHead(self.config)
    
    def forward(
        self,
        dataset_idx: int,
        input_ids: Optional[torch.Tensor] = None,
        input_scales: Optional[torch.Tensor] = None,
        feature_cls_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        tail_prompt: Optional[torch.Tensor] = None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.tpberta(
            input_ids,
            input_scales=input_scales,
            feature_cls_mask=feature_cls_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            tail_prompt=tail_prompt,
        )
        sequence_output = outputs[0]
        
        if tail_prompt is None: # ordinary finetune
            logits = self.heads[dataset_idx](sequence_output)
        else: # prompt-based tunning (not used in the paper)
            logits = self.mlm_head(sequence_output)[:, 0, :] # b, v; [cls] output

        if self.mlm and hasattr(self, 'mlm_head') and self.training: # for mlm task
            masked_input_ids, masked_tokens, numerical_mask, n_numerical_tokens \
                = mask_inputs(input_ids, token_type_ids)
            if n_numerical_tokens == 0: # no numerical features
                return logits.squeeze(1), (outputs,)
            masked_output = self.tpberta(
                masked_input_ids,
                input_scales=input_scales,
                feature_cls_mask=feature_cls_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )[0]
            masked_logits = self.mlm_head(masked_output) # b, f, v
            numerical_logits = masked_logits[:, 1:1+n_numerical_tokens] # b, n_num, v, exclude cls
            masked_numerical_logits = numerical_logits.reshape(
                -1, self.config.vocab_size
            )[numerical_mask]
            mlm_loss = nn.functional.cross_entropy(masked_numerical_logits, masked_tokens, reduction='none')
            return logits.squeeze(1), (outputs, mlm_loss)
        return logits.squeeze(1), outputs # don't use unsqueeze() for parallel training ! if [2, 3] -> [1, 3] x 2 -> [3] x 2 -> [6]


class TPBertaForClassification(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]
    base_model_prefix = 'tpberta'

    class TPBertaRankHead(nn.Module):
        """A ranker head for triplet loss"""
        def __init__(self, config) -> None:
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        def forward(self, x):
            x = self.dense(x)
            x = gelu(x)
            x = self.layer_norm(x)
            return x

    def __init__(self, config, num_class):
        super().__init__(config)

        self.num_labels = num_class
        self.config = config
        self.tpberta = TPBerta(config, add_pooling_layer=False)
        self.classifier = TPBertaHead(config, num_class)
        self.ranker = self.TPBertaRankHead(config)
    
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_scales: Optional[torch.Tensor] = None,
        feature_cls_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.tpberta(
            input_ids,
            input_scales=input_scales,
            feature_cls_mask=feature_cls_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]
        logits = self.classifier(sequence_output)

        return logits.squeeze(1), outputs

def build_default_model(args, data_cfg, num_classes: Union[List[int], int], device, pretrain=True, initialization=None):
    if pretrain and isinstance(num_classes, list): # MTL pre-training
        config = RobertaConfig.from_pretrained(Path(args.base_model_dir))
        model = TPBertaForMTLPretrain.from_pretrained(Path(args.base_model_dir), config=config, num_classes=num_classes, add_mlm_head=getattr(args, 'mlm', False), prompt_based=getattr(args, 'prompt', False)) 
        model.resize_token_embeddings(len(data_cfg.tokenizer))
        model.resize_mlm_head()
        
        setattr(config, 'max_position_embeddings', args.max_position_embeddings)
        setattr(config, 'type_vocab_size', args.type_vocab_size)

        # resize postion and type embeddings
        base_model = model.base_model
        base_model.embeddings.position_embeddings = \
            base_model._get_resized_embeddings(base_model.embeddings.position_embeddings, config.max_position_embeddings)
        base_model.embeddings.token_type_embeddings = \
            base_model._get_resized_embeddings(base_model.embeddings.token_type_embeddings, config.type_vocab_size)
        
        setattr(model, 'config', config)
    else:
        config = RobertaConfig.from_pretrained(args.pretrain_dir)
        if pretrain: # use pretrain params
            # choose to load last or best model
            try:
                model = TPBertaForClassification.from_pretrained(Path(args.pretrain_dir) / args.model_suffix, config=config, num_class=num_classes)
            except RuntimeError: # mismatch with postion ids (forget process in pre-train)
                print('try to fix mismatching in from_pretrained')
                state_file = Path(args.pretrain_dir) / args.model_suffix / 'pytorch_model.bin'
                state_dict = torch.load(state_file, map_location='cpu')
                state_dict['tpberta.embeddings.position_ids'] = \
                    state_dict['tpberta.embeddings.position_ids'][:,:config.max_position_embeddings]
                torch.save(state_dict, state_file)
                model = TPBertaForClassification.from_pretrained(Path(args.pretrain_dir) / args.model_suffix, config=config, num_class=num_classes)
        else:
            assert initialization in ['random', 'lm'] # initialized randomly or using pre-trained LM weights
            if initialization == 'random':
                model = TPBertaForClassification(config, num_class=num_classes) # init randomly
            else:
                assert hasattr(args, 'base_model')
                _config = RobertaConfig.from_pretrained(args.base_model)
                model = TPBertaForClassification.from_pretrained(args.base_model, config=_config, num_class=num_classes) 
                model.resize_token_embeddings(len(data_cfg.tokenizer))

                setattr(_config, 'max_position_embeddings', config.max_position_embeddings)
                setattr(_config, 'type_vocab_size', config.type_vocab_size)

                # resize postion and type embeddings
                base_model = model.base_model
                base_model.embeddings.position_embeddings = \
                    base_model._get_resized_embeddings(base_model.embeddings.position_embeddings, _config.max_position_embeddings)
                base_model.embeddings.token_type_embeddings = \
                    base_model._get_resized_embeddings(base_model.embeddings.token_type_embeddings, _config.type_vocab_size)
                
                setattr(model, 'config', _config)
                config = _config
                
    if device.type != 'cpu':
        model = model.to(device)
        if torch.cuda.device_count() > 1:  # type: ignore[code]
            print('Using nn.DataParallel')
            model = nn.DataParallel(model)
    
    return config, model