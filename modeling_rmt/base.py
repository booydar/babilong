import math
import torch
import torch.nn.functional as F

class RMTBaseModel(torch.nn.Module):
    def __init__(self, base_model, **rmt_kwargs):
        super().__init__()
        self.model = base_model
        self.set_params(**rmt_kwargs)

    def set_params(self, num_mem_tokens, tokenizer, **rmt_config):
        self.rmt_config = rmt_config
        self.extract_special_tokens(tokenizer)
        self.extend_word_embeddings(num_mem_tokens, tokenizer)

        self.segment_size = rmt_config['input_size'] - num_mem_tokens - tokenizer.num_special_tokens_to_add()
        if 'sep_token' in tokenizer.special_tokens_map:
            self.segment_size -= 1

    def set_memory(self, input_shape):
        memory = self.model.embeddings(self.mem_token_ids)
        memory = memory.repeat(input_shape[0], 1, 1)
        return memory

    def extract_special_tokens(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id
        self.special_token_ids = [tokenizer.pad_token_id]
        for token in ['cls_token', 'sep_token', 'eos_token', 'bos_token']:
            token_id = getattr(tokenizer, f'{token}_id')
            if token_id is not None:
                self.register_buffer(token, torch.tensor([token_id]))
                self.special_token_ids.append(token_id)
            else:
                setattr(self, token, None)

    def extend_word_embeddings(self, num_mem_tokens, tokenizer):
            
        vocab_size = self.model.config.vocab_size
        extended_vocab_size = vocab_size + num_mem_tokens
        self.num_mem_tokens = num_mem_tokens
        self.register_buffer('mem_token_ids', torch.arange(vocab_size, vocab_size + num_mem_tokens))
        self.model.resize_token_embeddings(extended_vocab_size)

        special_tokens = tokenizer.special_tokens_map
        mem_start_ind = int('cls_token' in special_tokens or 'bos_token' in special_tokens)
        self.memory_position = range(mem_start_ind, mem_start_ind + num_mem_tokens)
        self.model.embeddings = self.model.get_input_embeddings()

    def forward(self, **kwargs):
       raise NotImplementedError

    def pad_and_segment(self, input_ids):
        segmented_batch = []
        for seq in input_ids:
            drop_mask = torch.any(torch.stack([seq == t for t in self.special_token_ids if t is not None]), dim=0)
            seq = seq[~drop_mask]
            seq = seq[:self.segment_size * self.rmt_config['max_n_segments']]

            align = self.rmt_config.get('segment_alignment')
            if align in {'right', None}:
                split_inds = (list(range(len(seq), 0, -self.segment_size)) + [0])[::-1]
            elif align == 'left':
                split_inds = list(range(0, len(seq), self.segment_size)) + [len(seq)]
            elif align == 'center':
                n_seg = math.ceil(len(seq) / self.segment_size)
                split_inds = list(range(0, len(seq), math.ceil(len(seq) / n_seg))) + [len(seq)]
            else:
                raise NotImplementedError

            input_segments = [seq[start:end] for (start, end) in zip(split_inds, split_inds[1:])]
            input_segments = [self.pad_add_special_tokens(t, self.rmt_config['input_size']) for t in input_segments]

            # add empty segment markers if needed
            n_empty_segments = self.rmt_config['max_n_segments'] - len(input_segments)
            input_segments = [None] * n_empty_segments + input_segments

            segmented_batch.append(input_segments)

        segmented_batch = [[sample[seg_num] for sample in segmented_batch] \
                            for seg_num in range(self.rmt_config['max_n_segments'])]
        return segmented_batch

    def pad_add_special_tokens(self, **kwargs):
        raise NotImplementedError

    def prepare_kwargs(self, segment_input_ids, kwargs):
        seg_kwargs = dict(**kwargs)
        non_empty_mask = [s is not None for s in segment_input_ids]
        if sum(non_empty_mask) == 0:
            return None, non_empty_mask
            
        input_ids = torch.stack([s for s in segment_input_ids if s is not None])
        inputs_embeds = self.model.embeddings(input_ids)

        seg_kwargs['input_ids'] = None
        seg_kwargs['inputs_embeds'] = inputs_embeds
        if seg_kwargs.get('labels') is not None:
            seg_kwargs['labels'] = seg_kwargs['labels'][non_empty_mask]
        seg_kwargs['attention_mask'] = self.get_attention_mask(input_ids)
        if seg_kwargs.get('token_type_ids') is not None:
            seg_kwargs['token_type_ids'] = self.get_token_type_ids(input_ids)
        seg_kwargs['output_hidden_states'] = True

        return seg_kwargs, non_empty_mask

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
            extracted['loss'] = torch.stack(losses).mean(dim=0)

        for key, value in extracted.items():
            rmt_out[key] = value
        
        # drop unnecessary hiddens to save memory
        if not output_hidden_states:
            for key in rmt_out.keys():
                if 'hidden_state' in key:
                    rmt_out[key] = None

        return rmt_out 
        
    def get_token_type_ids(self, tensor):
        return torch.zeros_like(tensor)

    def get_attention_mask(self, tensor):
        mask = torch.ones_like(tensor)
        mask[tensor == self.pad_token_id] = 0
        return mask