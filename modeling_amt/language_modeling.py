# copied from original ARTM repo: https://raw.githubusercontent.com/RodkinIvan/associative-recurrent-memory-transformer/refs/heads/framework_accel/modeling_amt/language_modeling.py
import math
import torch
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from torch.nn.functional import relu as r

def dpfp(x, nu=1):
  x = torch.cat([r(x), r(-x)], dim=-1)
  x_rolled = torch.cat([x.roll(shifts=j, dims=-1)
           for j in range(1,nu+1)], dim=-1)
  x_repeat = torch.cat([x] * nu, dim=-1)
  return x_repeat * x_rolled

class DPFP:
    def __init__(self, nu):
        self.nu = nu
    
    def __call__(self, x):
        nu = self.nu
        x = torch.cat([r(x), r(-x)], dim=-1)
        x_rolled = torch.cat([x.roll(shifts=j, dims=-1) for j in range(1,nu+1)], dim=-1)
        x_repeat = torch.cat([x] * nu, dim=-1)
        return x_repeat * x_rolled

class AssociativeLayerWrapper(torch.nn.Module):

    def __init__(self, layer, d_model, num_mem_tokens, d_mem, correction=True, info=None) -> None:
        super().__init__()
        self.info = info
        self.seg_num = 0
        self.d_model = d_model
        self.num_mem_tokens = num_mem_tokens
        self.d_mem = d_mem

        nu = 3
        self.d_key = 2 * nu * d_mem
        self.phi = DPFP(nu)
        # self.d_key = d_mem
        # self.phi = torch.nn.Identity()

        self.W_mq = torch.nn.Linear(d_model, d_mem, bias=False, dtype=torch.bfloat16)
        # torch.nn.init.zeros_(self.W_mq.weight)
        self.W_mk = torch.nn.Linear(d_model, d_mem, bias=False, dtype=torch.bfloat16)
        self.W_mv = torch.nn.Linear(d_model, d_model, bias=False, dtype=torch.bfloat16)
        torch.nn.init.zeros_(self.W_mv.weight)
        self.W_mb = torch.nn.Linear(d_model, 1, dtype=torch.bfloat16)

        self.W_mem = torch.zeros(1, self.d_key, d_model, dtype=torch.bfloat16)
        self.z = torch.zeros(1, self.d_key, dtype=torch.bfloat16)
        self.W_mem.requires_grad_(False)
        self.z.requires_grad_(False)
        
        # self.ln = torch.nn.LayerNorm(d_model)

        self.zero_mem()
    
        self.layer = layer
        
        self.generate_mode = False
        self.first_seg = True
        self.correction = correction

    def associate(self, hidden_states):

        self.W_mem = self.W_mem.to(hidden_states.device).to(torch.bfloat16)
        self.z = self.z.to(hidden_states.device).to(torch.bfloat16)
        
        mq = self.phi(self.W_mq(hidden_states)).to(torch.bfloat16) # (bsz, seq_len, 2d_mem * nu)

        # crutch for dataparallel
        # mq += 0 * self.W_mb(hidden_states).sum() * self.W_mk(hidden_states).sum() * self.W_mv(hidden_states).sum() 
        #print(mq, self.W_mem)
        #print(mq.dtype, self.W_mem.dtype)
        num = torch.einsum('ijk,ikt->ijt', mq, self.W_mem)
        denom = torch.einsum("ik,ijk->ij", self.z, mq)[..., None] + 1e-5
        hidden_states = num / denom

        return hidden_states
    
    def forward(self, hidden_states, **kwargs):
        if not self.first_seg:
            hidden_states = self.associate(
                # self.ln(
                    hidden_states
                # )
            ) + hidden_states
        out = self.layer(hidden_states=hidden_states, **kwargs)
        if not self.generate_mode:
            mem_tokens = out[0][:, -self.num_mem_tokens:]
            self.update_mem(mem_tokens)
            self.first_seg = False
        return out

    def update_mem(self, mem_tokens):

        self.W_mem = self.W_mem.to(mem_tokens.device)
        self.z = self.z.to(mem_tokens.device)

        mk = self.phi(self.W_mk(mem_tokens))
        new_mv = self.W_mv(mem_tokens) # (bsz, num_mem_tokens, d_model)
        if not self.first_seg:
            num = torch.einsum('ijk,ikt->ijt', mk, self.W_mem)
            denom = torch.einsum("ij,ikj->ik", self.z, mk)[..., None] + 1e-5
            prev_mv = num / denom
            if self.correction:
                new_info_coef = 1 - denom / (torch.linalg.norm(mk, dim=-1) ** 2 + 1e-5)[..., None]
                new_info_coef = torch.clip(new_info_coef, 0, 1).detach()
            else:
                new_info_coef = 1
        else: 
            prev_mv = torch.zeros_like(new_mv, device=new_mv.device)
            new_info_coef = 1
        
        # wandb.log({f"gamma_{self.info['layer']}": new_info_coef.mean(dim=1).item() if isinstance(new_info_coef, torch.Tensor) else 1}, step=self.seg_num)
        mv = new_mv - prev_mv

        # new_norm = torch.linalg.norm(new_mv, dim=-1)
        # old_norm = torch.linalg.norm(prev_mv, dim=-1)
        # new_info_coef = torch.clip(1 - old_norm / (new_norm + 1e-5), -10, 10)[..., None].detach()
        # new_info_coef = 1 - denom

        mb = torch.sigmoid(self.W_mb(mem_tokens))[..., 0]

        associations =  torch.einsum('ijk,ijt,ij->ikt', mk, mv, mb) # (bsz, d_mem, d_model)
        self.W_mem = self.W_mem + associations

        self.z = self.z + (new_info_coef*mk).sum(dim=1)
        # self.z = self.z + (new_info_coef*mb[..., None]*mk).sum(dim=1)
        self.seg_num += 1


    def zero_mem(self):
        self.first_seg = True
        self.W_mem = torch.zeros(1, self.d_key, self.d_model)
        self.z = torch.zeros(1, self.d_key)
        self.seg_num = 0



class AssociativeMemoryCell(torch.nn.Module):
    def __init__(self, base_model, num_mem_tokens, d_mem, layers_attr: str = 'transformer.h', wrap_pos=True, correction=True, use_lora=False):
        super().__init__()
        self.model = base_model
        self.num_mem_tokens = num_mem_tokens
        self.d_mem = d_mem
        self.d_model = base_model.get_input_embeddings().embedding_dim
        self.W_mq = torch.nn.ModuleList()
        self.W_mem = []
        if use_lora:
            # LoRA case
            self.layers = self.model.model
        else:
            self.layers = self.model

        self.layers_attrs = layers_attr.split('.')
        for i, attr in enumerate(self.layers_attrs):
            self.layers = getattr(self.layers, attr)
        
        for i in range(len(self.layers)):
            self.layers[i] = AssociativeLayerWrapper(
                self.layers[i], 
                self.d_model, 
                self.num_mem_tokens, 
                self.d_mem, 
                correction,
                info={'layer': i}
            )
        self.create_memory(num_mem_tokens)
        self.wrap_pos = wrap_pos
        if wrap_pos:
            self.wrap_positional_embeddings(num_mem_tokens)
    
    def generate_mode(self, is_on):
        for layer in self.layers:
            layer.generate_mode = is_on
    
    def create_memory(self, num_mem_tokens):
        self.num_mem_tokens = num_mem_tokens
        embeddings = self.model.get_input_embeddings()
        memory_dim =  getattr(self.model.config, 'n_embd', self.model.config.hidden_size)
        memory_weights = torch.randn((num_mem_tokens, memory_dim)) * embeddings.weight.data.std()
        self.register_parameter('memory', torch.nn.Parameter(memory_weights, requires_grad=True))

    def wrap_positional_embeddings(self, num_mem_tokens):
        num_pos_embs, emb_dim = self.model.transformer.wpe.weight.shape
        prev_embs = self.model.transformer.wpe.weight.detach()
        self.model.transformer.wpe = torch.nn.Embedding(num_mem_tokens + num_pos_embs, emb_dim)

        new_num_pos = num_pos_embs + num_mem_tokens
        with torch.no_grad():
            self.model.transformer.wpe.weight[:len(self.model.transformer.wpe.weight)-num_mem_tokens] = prev_embs
        for layer in self.model.transformer.h:
            layer.layer.attn.bias = torch.tril(torch.ones((new_num_pos, new_num_pos), dtype=torch.uint8)).view(
                1, 1, new_num_pos, new_num_pos
            )

    def set_memory(self, input_shape):
        memory = self.memory.repeat(input_shape[0], 1, 1)
        return memory

    def zero_mem(self):
        for layer in self.layers:
            layer.zero_mem()

    def forward(self, input_ids, labels=None, labels_mask=None, zero_mem=False, **kwargs):
        if zero_mem:
            self.zero_mem()


        seg_kwargs = self.process_input(input_ids, **kwargs)

        out = self.model(**seg_kwargs)

        out = self.process_output(out, labels, labels_mask, **kwargs)

        return out

    def process_input(self, input_ids, **kwargs):
        memory_state = self.set_memory(input_ids.shape)
        seg_kwargs = dict(**kwargs)
        inputs_embeds = kwargs.get('inputs_embeds')
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([inputs_embeds, memory_state], dim=1)

        seg_kwargs['input_ids'] = None
        seg_kwargs['inputs_embeds'] = inputs_embeds
        if kwargs.get('attention_mask') is not None:
            seg_kwargs['attention_mask'] = self.pad_attention_mask(kwargs['attention_mask'], inputs_embeds.shape)
            if kwargs.get('prev_attn_mask') is not None:
                seg_kwargs['attention_mask'] = torch.cat([kwargs['prev_attn_mask'], seg_kwargs['attention_mask']], dim=-1)
        if 'prev_attn_mask' in seg_kwargs.keys():
            seg_kwargs.pop('prev_attn_mask')
        seg_kwargs['output_hidden_states'] = True

        if self.wrap_pos:
            num_pos_embs = self.model.transformer.wpe.weight.shape[0]
            ordinary_pos = torch.arange(0, input_ids.size(1), dtype=torch.long, device=input_ids.device)
            write_pos = torch.arange(num_pos_embs - self.num_mem_tokens, num_pos_embs, dtype=torch.long, device=input_ids.device)
            seg_kwargs['position_ids'] = torch.cat([
                ordinary_pos, 
                write_pos
            ]).long().unsqueeze(0)
        return seg_kwargs
    
    def pad_attention_mask(self, attention_mask, shape):
        if self.num_mem_tokens in {0, None}:
            return attention_mask
        else:
            mask = torch.ones(*shape[:2], dtype=torch.int64).to(attention_mask.device)
            mask[:, :-self.num_mem_tokens] = attention_mask
            return mask
    
    def process_output(self, model_outputs, labels, labels_mask, **kwargs):
        if self.num_mem_tokens not in {0, None}:
            out = CausalLMOutputWithCrossAttentions()
            out['logits'] = model_outputs.logits[:, :-self.num_mem_tokens]
            if kwargs.get('output_hidden_states'):
                out['hidden_states'] = [lh[:, :-self.num_mem_tokens] for lh in model_outputs.hidden_states]
            if kwargs.get('output_attentions'):
                out['attentions'] = model_outputs['attentions']
        else:
            out = model_outputs

        if labels is not None:
            ce_loss_fn = CrossEntropyLoss()
            logits = out['logits'][..., :-1, :].contiguous()
            flat_logits = logits.view(-1, logits.size(-1))
            labels = labels[..., 1:].contiguous()
            flat_labels = labels.view(-1)
            if labels_mask is not None:
                flat_mask = labels_mask[..., :-1].contiguous().view(-1)

                flat_logits = flat_logits[flat_mask]
                flat_labels = flat_labels[flat_mask]
            ce_loss = ce_loss_fn(flat_logits, flat_labels)
            out['ce_loss'] = ce_loss

        if kwargs.get('use_cache') is not None:
            out['past_key_values'] = model_outputs.past_key_values
        
        return out
    
    def generate(self, input_ids, attention_mask, zero_mem=False, **generate_kwargs):
        if zero_mem:
            self.zero_mem()
        
        
        self.generate_mode(True)
        seg_kwargs = self.process_input(input_ids, attention_mask=attention_mask)
        out = self.model.generate(
            inputs_embeds=seg_kwargs['inputs_embeds'][:, :-self.num_mem_tokens], 
            attention_mask=seg_kwargs['attention_mask'][:, :-self.num_mem_tokens], 
            **generate_kwargs
        )
        self.generate_mode(False)
        return out
    

class AssociativeRecurrentWrapper(torch.nn.Module):
    def __init__(self, memory_cell, **rmt_kwargs):
        super().__init__()
        
        self.memory_cell = memory_cell
        self.rmt_config = rmt_kwargs

    def forward(self, 
                input_ids, 
                labels=None, 
                labels_mask=None, 
                inputs_embeds=None, 
                attention_mask=None, 
                output_attentions=None, 
                output_hidden_states=None,
                input_segmented=False,
                sliding_window=False,
                ):
        if input_segmented:
            n_segs = input_ids.shape[1] if not (input_ids is None) else inputs_embeds.shape[1]
            segmented = [dict(
                input_ids=input_ids[:, i] if not (input_ids is None) else None, 
                inputs_embeds=inputs_embeds[:, i] if not (inputs_embeds is None) else None, 
                attention_mask=attention_mask[:, i],
                labels=labels[:, i] if not (labels is None) else None, 
                labels_mask=labels_mask[:, i] if not (labels_mask is None) else None, 
            ) for i in range(n_segs)]
            labels = torch.cat([labels[:, i] for i in range(n_segs)], dim=1)
            if labels_mask is not None:
                labels_mask = torch.cat([labels_mask[:, i] for i in range(n_segs)], dim=1)
        else:
            segmented = self.segment(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels, labels_mask=labels_mask)
        cell_outputs = []
        past_key_values = None
        num_mem_tokens = self.memory_cell.num_mem_tokens
        prev_attn_mask = None
        self.memory_cell.zero_mem()
        for seg_num, segment in enumerate(segmented):
            seg_len = segment['input_ids'].size(-1)
            cell_out = self.memory_cell(**segment,  
                                        output_hidden_states=True, 
                                        use_cache=sliding_window, 
                                        past_key_values=past_key_values,
                                        prev_attn_mask=prev_attn_mask,
                                        zero_mem=False
            )
            if sliding_window:
                prev_attn_mask = segment['attention_mask']
                past_key_values = [
                    [
                        k_or_v[..., -(num_mem_tokens+seg_len):k_or_v.size(-2)-num_mem_tokens, :].detach() 
                        for k_or_v in seg_kv
                    ] 
                    for seg_kv in cell_out['past_key_values']
                ]
            cell_outputs.append(cell_out)
        self.memory_cell.zero_mem()


        out = self.process_outputs(cell_outputs, labels=labels, 
                                   labels_mask=labels_mask,
                                   output_attentions=output_attentions, 
                                   output_hidden_states=output_hidden_states)
        return out

    def segment(self, **kwargs):
        segments = []
        for k, tensor in kwargs.items():
            if tensor is not None:
                k_segments = self.split_tensor(tensor)
                for s, k_seg in enumerate(k_segments):
                    if s < len(segments):
                        segments[s][k] = k_seg
                    else:
                        segments.append({k: k_seg})

        if len(segments) > self.rmt_config.get('max_n_segments', torch.inf):
            raise ValueError(f"Too many segments in input: {len(segments)}")
        return segments
    
    def split_tensor(self, tensor):
        align = self.rmt_config.get('segment_alignment')
        segment_size = self.rmt_config.get('segment_size')
        if align in {'left', None}:
            split_inds = list(range(0, tensor.shape[1], segment_size)) + [tensor.shape[1]]
            segments = [tensor[:, start:end] for (start, end) in zip(split_inds, split_inds[1:])]
        elif align in {'right', None}:
            split_inds = (list(range(tensor.shape[1], 0, -segment_size)) + [0])[::-1]
            segments = [tensor[:, start:end] for (start, end) in zip(split_inds, split_inds[1:])]
        elif align == 'center':
            n_seg = math.ceil(tensor.shape[1] / segment_size)
            segments = torch.chunk(tensor, n_seg, dim=1)
        else:
            raise NotImplementedError
        return segments

    def process_outputs(self, cell_outputs, **kwargs):
        out = CausalLMOutputWithCrossAttentions()
        full_logits = torch.cat([o.logits for o in cell_outputs], dim=1)
        full_hidden_states = tuple([torch.cat(layer_hs, dim=1) for layer_hs in zip(*[o.hidden_states for o in cell_outputs])])

        labels = kwargs.get('labels')
        if labels is not None:
            shift_labels = labels[..., 1:].contiguous()
            shift_logits = full_logits[..., :-1, :].contiguous()
            flat_labels = shift_labels.view(-1)
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            
            loss_fct = CrossEntropyLoss()
            labels_mask = kwargs.get('labels_mask')
            if labels_mask is not None:
                shift_mask = labels_mask[..., :-1].contiguous()

                flat_labels = flat_labels[shift_mask.view(-1)]
                flat_logits = flat_logits[shift_mask.view(-1)]
                
            out['loss'] = loss_fct(flat_logits, flat_labels)
        else:
            out['loss'] = 0 

        if self.rmt_config.get("return_all_logits", False):
            out['ce_loss'] = out['loss']
        
        out['logits'] = full_logits
        segment_keys = ['loss', 'logits']
        if kwargs.get('output_attentions'):
            segment_keys.append('attentions')
        if kwargs.get('output_hidden_states'):
            segment_keys.append('hidden_states')
            out['hidden_states'] = full_hidden_states

        if self.rmt_config.get("return_all_logits", False):
            for seg_num, o in enumerate(cell_outputs):
                for key, value in o.items():
                    if any([sk in key for sk in segment_keys]):
                        out[f'{key}_{seg_num}'] = value
        return out 
        
    def manage_gradients(self, memory_state, seg_num):
        k2, max_n_segments = self.rmt_config.get('k2'), self.rmt_config.get('max_n_segments')
        if seg_num == 0 \
            or k2 in {-1, None} \
            or seg_num + k2 > max_n_segments:
                return True
        
        memory_state = memory_state.detach()
        return False
    
    def generate(self, input_ids, attention_mask, **generate_kwargs):
        self.memory_cell.zero_mem()
        segmented = self.segment(input_ids=input_ids, attention_mask=attention_mask)

        for seg_num, segment in enumerate(segmented[:-1]):
            cell_out = self.memory_cell(**segment, output_hidden_states=True, zero_mem=False)

        final_segment = segmented[-1]
        out = self.memory_cell.generate(**final_segment, zero_mem=False, **generate_kwargs)
        self.memory_cell.zero_mem()
        return out

    def gradient_checkpointing_enable(self, *args, **kwargs):
        # doesn't supported for ARMT
        self.memory_cell.model.gradient_checkpointing_enable(*args, **kwargs)