import torch
import torch.nn.functional as F
from .base import RMTBaseModel

class RMTEncoderDecoderForConditionalGeneration(RMTBaseModel):
    def forward(self, input_ids, attention_mask=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        kwargs = {'attention_mask': attention_mask,
                #   'position_ids': position_ids, 
                  'inputs_embeds': inputs_embeds,
                  'labels': labels, 'output_attentions': output_attentions,
                  'output_hidden_states': output_hidden_states, 'return_dict': return_dict,
                  }

        memory = self.set_memory(input_ids.shape)
        segmented = self.pad_and_segment(input_ids)

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
            
            memory[non_empty_mask] = out.encoder_hidden_states[-1][:, self.memory_position]

        out = self.process_outputs(base_model_outputs, output_attentions, output_hidden_states)
        return out

    def generate(self, input_ids, attention_mask=None, position_ids=None, head_mask=None,
                inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None,
                min_length=None, max_length=None):
        kwargs = {'attention_mask': attention_mask,
                  'inputs_embeds': inputs_embeds,
                  'output_attentions': output_attentions,
                  'output_hidden_states': output_hidden_states, 'return_dict': return_dict,
                  'min_length': min_length, 'max_length': max_length
                  }

        memory = self.set_memory(input_ids.shape)
        segmented = self.pad_and_segment(input_ids)

        base_model_outputs = []
        for seg_num, segment_input_ids in enumerate(segmented):                
            if self.rmt_config['bptt_depth'] != -1:
                raise NotImplementedError

            seg_kwargs, non_empty_mask = self.prepare_kwargs(segment_input_ids, kwargs)
            if sum(non_empty_mask) == 0:
                continue
            seg_kwargs['inputs_embeds'][:, self.memory_position] = memory[non_empty_mask]
            if seg_num == len(segmented) - 1:
                out = self.model.generate(**seg_kwargs)
            else:
                for param in ['min_length', 'max_length']:
                    if param in seg_kwargs:
                        seg_kwargs.pop(param)
                        
                out = self.model.encoder(**seg_kwargs)
                memory[non_empty_mask] = out.last_hidden_state[:, self.memory_position]
                # base_model_outputs.append(out)

        return out

    def pad_add_special_tokens(self, tensor, segment_size):
        input_elements = []
        if self.bos_token is not None:
            input_elements.append(self.bos_token)
        input_elements += [self.mem_token_ids, tensor, self.eos_token]
        tensor = torch.cat(input_elements)

        pad_size = segment_size - tensor.shape[0]
        if pad_size > 0:
            tensor = F.pad(tensor, (0, pad_size), value=self.pad_token_id)
        return tensor


import types
class RMTEncoderDecoderMemoryLayers(RMTEncoderDecoderForConditionalGeneration):
    def set_params(self, num_mem_tokens, tokenizer, **rmt_config):
        super().set_params(num_mem_tokens, tokenizer, **rmt_config)
        self.add_memory_layers()
        self.override_encoder_forward(rmt_config.get('memory_forward_func'))

    def override_encoder_forward(self, memory_forward_func):
        if self.rmt_config.get('memory_layers') is None:
            return
        if memory_forward_func is None:
            from rmt_utils.encoder_decoder.memory_layers import memory_layers_forward
            memory_forward_func = memory_layers_forward
        encoder_forward = lambda *args, **kwargs: memory_forward_func(*args, **kwargs, rmt_parent=self)
        self.model.base_model.encoder.forward = types.MethodType(encoder_forward, self.model.base_model.encoder)

    def add_memory_layers(self):
        memory_layers, share_memory_layers = self.rmt_config.get('memory_layers'), self.rmt_config.get('share_memory_layers')
        if memory_layers is None:
            self.memory_layers = None
        else:
            if memory_layers == 'all':
                memory_layers = range(len(self.model.encoder.block))
            else:
                raise NotImplementedError
                
            if share_memory_layers:
                memory_layer = copy.deepcopy(self.model.encoder.block[0])
                self.memory_layers = [memory_layer for _ in range(len(memory_layers))]
                for n, p in memory_layer.named_parameters():
                    param_name = re.sub('\.', '_', f'memory_{n}')
                    self.register_parameter(param_name, p)
            else:
                self.memory_layers = [copy.deepcopy(self.model.encoder.block[int(l)]) for l in memory_layers]
                for ln, layer in enumerate(self.memory_layers):
                    for n, p in layer.named_parameters():
                        param_name = re.sub('\.', '_', f'{ln}_memory_{n}')
                        self.register_parameter(param_name, p)


import copy
import re
from rmt_utils.encoder_decoder.horizontal_memory import horizontal_memory_forward
class RMTEncoderDecoderHorizontalMemory(RMTEncoderDecoderMemoryLayers):
    def set_params(self, num_mem_tokens, tokenizer, **rmt_config):
        super().set_params(num_mem_tokens, tokenizer, **rmt_config)
        self.add_memory_layers()
        
        memory_forward_func = rmt_config.get('memory_forward_func')
        if not memory_forward_func:
            memory_forward_func = horizontal_memory_forward
        self.override_encoder_forward(memory_forward_func)

    def set_memory(self, input_shape):
        self.memory_storage = {}
        memory = self.model.embeddings(self.mem_token_ids)
        memory = memory.repeat(input_shape[0], 1, 1)
        
        # fill layer memories 
        memory_input = self.pad_add_special_tokens(self.mem_token_ids, self.num_mem_tokens)
        mem_out = self.model.encoder(memory_input.reshape((1, -1)))
        
        return memory

    def forward(self, input_ids, attention_mask=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        kwargs = {'attention_mask': attention_mask,
                #   'position_ids': position_ids, 
                  'inputs_embeds': inputs_embeds,
                  'labels': labels, 'output_attentions': output_attentions,
                  'output_hidden_states': output_hidden_states, 'return_dict': return_dict,
                  }

        memory = self.set_memory(input_ids.shape)
        segmented = self.pad_and_segment(input_ids)

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
            
            memory[non_empty_mask] = out.encoder_hidden_states[-1][:, self.memory_position]

        out = self.process_outputs(base_model_outputs, output_attentions, output_hidden_states)
        return out

    def generate(self, input_ids, attention_mask=None, position_ids=None, head_mask=None,
                inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None,
                min_length=None, max_length=None):
        kwargs = {'attention_mask': attention_mask,
                  'inputs_embeds': inputs_embeds,
                  'output_attentions': output_attentions,
                  'output_hidden_states': output_hidden_states, 'return_dict': return_dict,
                  'min_length': min_length, 'max_length': max_length
                  }

        memory = self.set_memory(input_ids.shape)
        segmented = self.pad_and_segment(input_ids)

        for seg_num, segment_input_ids in enumerate(segmented):                
            if self.rmt_config['bptt_depth'] != -1:
                raise NotImplementedError

            seg_kwargs, non_empty_mask = self.prepare_kwargs(segment_input_ids, kwargs)
            if sum(non_empty_mask) == 0:
                continue
            seg_kwargs['inputs_embeds'][:, self.memory_position] = memory[non_empty_mask]
            self.memory_storage['non_empty_mask'] = non_empty_mask

            if seg_num == len(segmented) - 1:
                out = self.model.generate(**seg_kwargs)
            else:
                for param in ['min_length', 'max_length']:
                    if param in seg_kwargs:
                        seg_kwargs.pop(param)
                        
                out = self.model.encoder(**seg_kwargs)
                memory[non_empty_mask] = out.last_hidden_state[:, self.memory_position]

        return out
    
from torch.nn import CrossEntropyLoss
class RMTEncoderDecoderMemoryOutput(RMTEncoderDecoderMemoryLayers):
    def forward(self, input_ids, attention_mask=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        kwargs = {'attention_mask': attention_mask,
                #   'position_ids': position_ids, 
                  'inputs_embeds': inputs_embeds,
                  'labels': labels, 'output_attentions': output_attentions,
                  'output_hidden_states': output_hidden_states, 'return_dict': return_dict,
                  }

        memory = self.set_memory(input_ids.shape)
        segmented = self.pad_and_segment(input_ids)

        memories = []
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
            
            memory[non_empty_mask] = out.encoder_hidden_states[-1][:, self.memory_position]
            memories.append(torch.clone(memory))

            if seg_num == len(segmented) - 1:
                memories = torch.cat(memories, dim=1)
                decoder_input_ids = self.model._shift_right(labels)

                decoder_outputs = self.model.decoder(input_ids=decoder_input_ids, encoder_hidden_states=memories)
                # base_model_outputs.append(out)

        out = self.process_outputs(base_model_outputs, output_attentions, output_hidden_states)

        sequence_output = decoder_outputs[0]
        # Set device for model parallelism
        if self.model.model_parallel:
            torch.cuda.set_device(self.model.encoder.first_device)
            self.model.lm_head = self.model.lm_head.to(self.model.encoder.first_device)
            sequence_output = sequence_output.to(self.model.lm_head.weight.device)

        if self.model.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model.model_dim**-0.5)

        lm_logits = self.model.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        
        out['loss'] = loss

        return out

    def generate(self, input_ids, attention_mask=None, position_ids=None, head_mask=None,
                inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None,
                min_length=None, max_length=None):
        kwargs = {'attention_mask': attention_mask,
                  'inputs_embeds': inputs_embeds,
                  'output_attentions': output_attentions,
                  'output_hidden_states': output_hidden_states, 'return_dict': return_dict,
                  'min_length': min_length, 'max_length': max_length
                  }

        memory = self.set_memory(input_ids.shape)
        segmented = self.pad_and_segment(input_ids)

        memories = []
        for seg_num, segment_input_ids in enumerate(segmented):                
            if self.rmt_config['bptt_depth'] != -1:
                raise NotImplementedError

            seg_kwargs, non_empty_mask = self.prepare_kwargs(segment_input_ids, kwargs)
            if sum(non_empty_mask) == 0:
                continue
            seg_kwargs['inputs_embeds'][:, self.memory_position] = memory[non_empty_mask]

            for param in ['min_length', 'max_length']:
                if param in seg_kwargs:
                    seg_kwargs.pop(param)
                    
            encoder_out = self.model.encoder(**seg_kwargs)
            memory[non_empty_mask] = out.last_hidden_state[:, self.memory_position]
            memories.append(torch.clone(memory))

            if seg_num == len(segmented) - 1:                
                memories = torch.cat(memories, dim=1)
                out = self.model.generate(**seg_kwargs, encoder_hidden_states=memories)

        hidden_states = torch.cat(memories, dim=1)
        encoder_out.hidden_states = None
        encoder_out.last_hidden_state = hidden_states
        out = self.model.generate(encoder_outputs=encoder_out)
        return out 