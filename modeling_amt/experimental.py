from modeling_amt.language_modeling import *

# class AssociativeMemoryCellLoRA(torch.nn.Module):
    # def __init__(self, base_model, num_mem_tokens, d_mem, layers_attr: str = 'transformer.h', wrap_pos=True, correction=True, use_lora=False):
    #     super().__init__()
    #     self.model = base_model
    #     self.num_mem_tokens = num_mem_tokens
    #     self.d_mem = d_mem
    #     self.d_model = base_model.get_input_embeddings().embedding_dim
    #     self.W_mq = torch.nn.ModuleList()
    #     self.W_mem = []
    #     if use_lora:
    #         # LoRA case
    #         self.layers = self.model.model
    #     else:
    #         self.layers = self.model

    #     self.layers_attrs = layers_attr.split('.')
    #     for i, attr in enumerate(self.layers_attrs):
    #         self.layers = getattr(self.layers, attr)
        
    #     for i in range(len(self.layers)):
    #         self.layers[i] = AssociativeLayerWrapper(
    #             self.layers[i], 
    #             self.d_model, 
    #             self.num_mem_tokens, 
    #             self.d_mem, 
    #             correction,
    #             info={'layer': i}
    #         )
    #     self.create_memory(num_mem_tokens)
    #     self.wrap_pos = wrap_pos
    #     if wrap_pos:
    #         self.wrap_positional_embeddings(num_mem_tokens)
    
    # def generate_mode(self, is_on):
    #     for layer in self.layers:
    #         layer.generate_mode = is_on
    
    # def create_memory(self, num_mem_tokens):
    #     self.num_mem_tokens = num_mem_tokens
    #     embeddings = self.model.get_input_embeddings()
    #     memory_dim =  getattr(self.model.config, 'n_embd', self.model.config.hidden_size)
    #     memory_weights = torch.randn((num_mem_tokens, memory_dim)) * embeddings.weight.data.std()
    #     self.register_parameter('memory', torch.nn.Parameter(memory_weights, requires_grad=True))

    # def wrap_positional_embeddings(self, num_mem_tokens):
    #     num_pos_embs, emb_dim = self.model.transformer.wpe.weight.shape
    #     prev_embs = self.model.transformer.wpe.weight.detach()
    #     self.model.transformer.wpe = torch.nn.Embedding(num_mem_tokens + num_pos_embs, emb_dim)

    #     new_num_pos = num_pos_embs + num_mem_tokens
    #     with torch.no_grad():
    #         self.model.transformer.wpe.weight[:len(self.model.transformer.wpe.weight)-num_mem_tokens] = prev_embs
    #     for layer in self.model.transformer.h:
    #         layer.layer.attn.bias = torch.tril(torch.ones((new_num_pos, new_num_pos), dtype=torch.uint8)).view(
    #             1, 1, new_num_pos, new_num_pos
    #         )

    # def set_memory(self, input_shape):
    #     memory = self.memory.repeat(input_shape[0], 1, 1)
    #     return memory

    # def zero_mem(self):
    #     for layer in self.layers:
    #         layer.zero_mem()

    # def forward(self, input_ids, labels=None, labels_mask=None, zero_mem=False, use_custom_lora=None, **kwargs):
    #     if zero_mem:
    #         self.zero_mem()


    #     seg_kwargs = self.process_input(input_ids, **kwargs)

    #     if use_custom_lora is None:
    #         out = self.model(**seg_kwargs)
    #     elif use_custom_lora:
    #         raise NotImplementedError("Using custom lora is not implemented")
    #     else:
    #         out = self.model(**seg_kwargs)

    #     out = self.process_output(out, labels, labels_mask, **kwargs)

    #     return out

class AssociativeRecurrentWrapperDisableLoRALS(AssociativeRecurrentWrapper):
    # def __call__(self, *args, **kwds):
    #     return self.forward(*args, **kwds)
    
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
        # print('in fwd')
        # 1/0
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
            # print("seg_num ", seg_num)
            if seg_num == len(segmented) - 1:
                self.memory_cell.model.disable_adapter_layers()
                # print('disabling adapter')
            seg_len = segment['input_ids'].size(-1)
            cell_out = self.memory_cell(**segment,  
                                        output_hidden_states=True, 
                                        use_cache=sliding_window, 
                                        past_key_values=past_key_values,
                                        prev_attn_mask=prev_attn_mask,
                                        zero_mem=False
            )
            if seg_num == len(segmented) - 1:
                self.memory_cell.model.enable_adapter_layers()
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
    
    def generate(self, input_ids, attention_mask, **generate_kwargs):
        # print('in gen')
        self.memory_cell.zero_mem()
        segmented = self.segment(input_ids=input_ids, attention_mask=attention_mask)

        for seg_num, segment in enumerate(segmented[:-1]):
            cell_out = self.memory_cell(**segment, output_hidden_states=True, zero_mem=False)

        final_segment = segmented[-1]
        self.memory_cell.model.disable_adapter_layers()
        out = self.memory_cell.generate(**final_segment, zero_mem=False, **generate_kwargs)
        self.memory_cell.model.enable_adapter_layers()
        self.memory_cell.zero_mem()
        return out