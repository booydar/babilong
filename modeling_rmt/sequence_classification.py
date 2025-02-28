import torch
import torch.nn.functional as F
from .base import RMTBaseModel

class RMTEncoderForSequenceClassification(RMTBaseModel):
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        kwargs = {'attention_mask': attention_mask, 'token_type_ids': token_type_ids,
                  'position_ids': position_ids, 'inputs_embeds': inputs_embeds,
                  'labels': labels, 'output_attentions': output_attentions,
                  'output_hidden_states': output_hidden_states, 'return_dict': return_dict,
                  }

        memory = self.set_memory(input_ids.shape)
        segmented = self.pad_and_segment(input_ids)
        # if self.num_mem_tokens == 0:
        #     segmented = segmented[-1:]

        base_model_outputs = []
        for seg_num, segment_input_ids in enumerate(segmented):                
            if self.rmt_config['bptt_depth'] != -1:
                raise NotImplementedError

            seg_kwargs, non_empty_mask = self.prepare_kwargs(segment_input_ids, kwargs)
            if sum(non_empty_mask) == 0:
                continue
            
            seg_kwargs['inputs_embeds'][:, self.memory_position] = memory[non_empty_mask]
            out = self.model(**seg_kwargs)
            base_model_outputs.append(out)
            
            memory[non_empty_mask] = out.hidden_states[-1][:, self.memory_position]

        out = self.process_outputs(base_model_outputs, output_attentions, output_hidden_states)
        return out

    def pad_add_special_tokens(self, tensor, segment_size):
        input_elements = []
        input_elements += [self.cls_token, self.mem_token_ids, self.sep_token, tensor, self.sep_token]
        tensor = torch.cat(input_elements)

        pad_size = segment_size - tensor.shape[0]
        if pad_size > 0:
            tensor = F.pad(tensor, (0, pad_size), value=self.pad_token_id)
        return tensor
        

import copy
import types
class RMTEncoderMemoryLayers(RMTEncoderForSequenceClassification):
    def set_params(self, num_mem_tokens, tokenizer, **rmt_config):
        super().set_params(num_mem_tokens, tokenizer, **rmt_config)
        self.add_memory_layers()
        self.override_encoder_forward(rmt_config.get('memory_forward_func'))

    def override_encoder_forward(self, memory_forward_func):
        if self.rmt_config.get('memory_layers') is None:
            return
        if memory_forward_func is None:
            from rmt_utils.encoder.memory_layers import memory_layers_forward
            memory_forward_func = memory_layers_forward
        encoder_forward = lambda *args, **kwargs: memory_forward_func(*args, **kwargs, rmt_parent=self)
        self.model.base_model.encoder.forward = types.MethodType(encoder_forward, self.model.base_model.encoder)

    def add_memory_layers(self):
        memory_layers, share_memory_layers = self.rmt_config.get('memory_layers'), self.rmt_config.get('share_memory_layers')
        if memory_layers is None:
            self.memory_layers = None
        else:
            if memory_layers == 'all':
                memory_layers = range(len(self.model.base_model.encoder.layer))
            else:
                raise NotImplementedError
                
            if share_memory_layers:
                memory_layer = copy.deepcopy(self.model.base_model.encoder.layer[0])
                self.memory_layers = [memory_layer for _ in range(len(memory_layers))]
                for n, p in memory_layer.named_parameters():
                    param_name = re.sub('\.', '_', f'memory_{n}')
                    self.register_parameter(param_name, p)
            else:
                self.memory_layers = [copy.deepcopy(self.model.base_model.encoder.layer[int(l)]) for l in memory_layers]
                for ln, layer in enumerate(self.memory_layers):
                    for n, p in layer.named_parameters():
                        param_name = re.sub('\.', '_', f'{ln}_memory_{n}')
                        self.register_parameter(param_name, p)


import re
import numpy as np
from torch.nn import CrossEntropyLoss
class RMTEncoderMLMMemLoss(RMTEncoderMemoryLayers):
    def set_params(self, num_mem_tokens, tokenizer, **rmt_config):
        super().set_params(num_mem_tokens, tokenizer, **rmt_config)
        self.add_reconstruction_layers()

    def add_reconstruction_layers(self):
        self.rec_attn = copy.deepcopy(self.model.base_model.encoder.layer[-1])
        self.rec_cls = torch.nn.Linear(self.model.config.hidden_size, self.model.config.vocab_size)

        for n, p in self.rec_attn.named_parameters():
            param_name = re.sub('\.', '_', f'rec_attn_{n}')
            self.register_buffer(param_name, p)
        
        for n, p in self.rec_cls.named_parameters():
            param_name = re.sub('\.', '_', f'rec_cls_{n}')
            self.register_parameter(param_name, p)
            
    def segment_reconstruction_forward(self, memory_outputs, previous_input_ids):
        mlm_prob = self.rmt_config['mlm_prob'] if 'mlm_prob' in self.rmt_config else 0.15
        
        input_embeddings = self.model.embeddings(previous_input_ids)
        input_embeddings[:, self.memory_position] = memory_outputs

        token_inds = list(range(self.num_mem_tokens + 2, input_embeddings.shape[1] - 1))
        mask_inds = np.random.choice(token_inds, round(len(token_inds) * mlm_prob))
        attention_mask = torch.ones(input_embeddings.shape[1]).to(device=input_embeddings.device)
        attention_mask[mask_inds] = 0

        rec_attn_out = self.rec_attn(input_embeddings, attention_mask=attention_mask)
        rec_logits = self.rec_cls(rec_attn_out[0])

        loss_fct = CrossEntropyLoss(ignore_index=-100)
        reconstruction_loss = loss_fct(rec_logits.view(-1, rec_logits.size(-1)), previous_input_ids.view(-1))
        
        return reconstruction_loss

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        kwargs = {'attention_mask': attention_mask, 'token_type_ids': token_type_ids,
                  'position_ids': position_ids, 'inputs_embeds': inputs_embeds,
                  'labels': labels, 'output_attentions': output_attentions,
                  'output_hidden_states': output_hidden_states, 'return_dict': return_dict,
                  }

        memory = self.set_memory(input_ids.shape)
        segmented = self.pad_and_segment(input_ids)
        if self.num_mem_tokens == 0:
            segmented = segmented[-1:]

        base_model_outputs = []
        for seg_num, segment_input_ids in enumerate(segmented):                
            if self.rmt_config['bptt_depth'] != -1:
                raise NotImplementedError

            seg_kwargs, non_empty_mask = self.prepare_kwargs(segment_input_ids, kwargs)
            if sum(non_empty_mask) == 0:
                continue
            
            seg_kwargs['inputs_embeds'][:, self.memory_position] = memory[non_empty_mask]
            out = self.model(**seg_kwargs)
            
            memory[non_empty_mask] = out.hidden_states[-1][:, self.memory_position]

            input_ids = torch.stack([s for s in segment_input_ids if s is not None])
            segment_reconstruction_loss = self.segment_reconstruction_forward(memory[non_empty_mask], input_ids)
            out['reconstruction_loss'] = segment_reconstruction_loss
            base_model_outputs.append(out)

        out = self.process_outputs(base_model_outputs, output_attentions, output_hidden_states)
        return out

    def process_outputs(self, model_outputs, output_attentions, output_hidden_states):
        rmt_out = model_outputs[-1]

        segment_keys = ['loss']
        if output_attentions:
            segment_keys.append('attentions')
        if output_hidden_states:
            segment_keys.append('hidden_states')

        extracted = {}
        for seg_num, out in enumerate(model_outputs):
            for key, value in out.items():
                if any([sk in key for sk in segment_keys]):
                    extracted[f'{key}_{seg_num}'] = value

        if self.rmt_config['sum_loss']:
            losses = [out['loss'] for out in model_outputs]
            extracted['loss'] = torch.stack(losses).mean()
        else:
            extracted['loss'] = rmt_out['loss']

        reconstruction_loss = torch.stack([out['reconstruction_loss'] for out in model_outputs]).mean()
        rec_coef = self.rmt_config['reconstruction_loss_coef']
        extracted['reconstruction_loss'] = reconstruction_loss
        extracted['loss'] =  reconstruction_loss * rec_coef + extracted['loss'] * (1 - rec_coef)
        

        for key, value in extracted.items():
            rmt_out[key] = value

        # drop unnecessary hiddens to save memory
        if not output_hidden_states:
            for key in rmt_out.keys():
                if 'hidden_state' in key:
                    rmt_out[key] = None

        return rmt_out 


from rmt_utils.encoder.horizontal_memory import horizontal_memory_forward
class RMTEncoderHorizontalMemory(RMTEncoderMemoryLayers):
    def set_params(self, num_mem_tokens, tokenizer, **rmt_config):
        rmt_config['input_size'] -= num_mem_tokens
        super().set_params(num_mem_tokens, tokenizer, **rmt_config)
        self.add_memory_layers()
        
        memory_forward_func = rmt_config.get('memory_forward_func')
        if memory_forward_func is None:
            memory_forward_func = horizontal_memory_forward
        self.override_encoder_forward(memory_forward_func)

    def set_memory(self, input_shape):
        self.memory_storage = {}
        memory = self.model.embeddings(self.mem_token_ids)
        memory = memory.repeat(input_shape[0], 1, 1)
        
        # fill layer memories 
        memory_input = self.pad_add_special_tokens(self.mem_token_ids, self.num_mem_tokens)
        mem_out = self.model(memory_input.reshape((1, -1)))
        
        return memory

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        kwargs = {'attention_mask': attention_mask, 'token_type_ids': token_type_ids,
                  'position_ids': position_ids, 'inputs_embeds': inputs_embeds,
                  'labels': labels, 'output_attentions': output_attentions,
                  'output_hidden_states': output_hidden_states, 'return_dict': return_dict,
                  }

        memory = self.set_memory(input_ids.shape)
        segmented = self.pad_and_segment(input_ids)
        if self.num_mem_tokens == 0:
            segmented = segmented[-1:]

        base_model_outputs = []
        for seg_num, segment_input_ids in enumerate(segmented):                
            if self.rmt_config['bptt_depth'] != -1:
                raise NotImplementedError

            seg_kwargs, non_empty_mask = self.prepare_kwargs(segment_input_ids, kwargs)
            if sum(non_empty_mask) == 0:
                continue
            self.memory_storage['non_empty_mask'] = non_empty_mask
            
            seg_kwargs['inputs_embeds'][:, self.memory_position] = memory[non_empty_mask]
            out = self.model(**seg_kwargs)
            base_model_outputs.append(out)
            
            memory[non_empty_mask] = out.hidden_states[-1][:, self.memory_position]

        out = self.process_outputs(base_model_outputs, output_attentions, output_hidden_states)
        return out