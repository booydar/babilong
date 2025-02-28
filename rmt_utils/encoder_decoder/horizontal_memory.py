import torch
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

def horizontal_memory_forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        rmt_parent=None,
    ):
    # Model parallel
    if self.model_parallel:
        torch.cuda.set_device(self.first_device)
        self.embed_tokens = self.embed_tokens.to(self.first_device)
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        err_msg_prefix = "decoder_" if self.is_decoder else ""
        raise ValueError(
            f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
        )
    elif input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        err_msg_prefix = "decoder_" if self.is_decoder else ""
        raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

    if inputs_embeds is None:
        assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
        inputs_embeds = self.embed_tokens(input_ids)

    batch_size, seq_length = input_shape

    # required mask seq length can be calculated via length of past
    mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length
    # mask_seq_length += rmt_parent.num_mem_tokens
    
    # print('mask_seq_length', mask_seq_length)

    if use_cache is True:
        assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

    if attention_mask is None:
        attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        
    
    memory_len = rmt_parent.num_mem_tokens
    memory_attention_mask = torch.ones_like(attention_mask)[:, :memory_len]
    attention_mask = torch.cat([memory_attention_mask, attention_mask], dim=-1)

    
    if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
        encoder_seq_length = encoder_hidden_states.shape[1]
        encoder_attention_mask = torch.ones(
            batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
        )

    # initialize past_key_values with `None` if past does not exist
    if past_key_values is None:
        past_key_values = [None] * len(self.block)

    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

    # If a 2D or 3D attention mask is provided for the cross-attention
    # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
    if self.is_decoder and encoder_hidden_states is not None:
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
        encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
    else:
        encoder_extended_attention_mask = None

    # Prepare head mask if needed
    head_mask = self.get_head_mask(head_mask, self.config.num_layers)
    cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
    present_key_value_states = () if use_cache else None
    all_hidden_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None
    all_cross_attentions = () if (output_attentions and self.is_decoder) else None
    position_bias = None
    encoder_decoder_position_bias = None

    hidden_states = self.dropout(inputs_embeds)

    for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
        layer_head_mask = head_mask[i]
        cross_attn_layer_head_mask = cross_attn_head_mask[i]
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(hidden_states.device)
            # Ensure that attention_mask is always on the same device as hidden_states
            if attention_mask is not None:
                attention_mask = attention_mask.to(hidden_states.device)
            if position_bias is not None:
                position_bias = position_bias.to(hidden_states.device)
            if encoder_hidden_states is not None:
                encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
            if encoder_extended_attention_mask is not None:
                encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
            if encoder_decoder_position_bias is not None:
                encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
            if layer_head_mask is not None:
                layer_head_mask = layer_head_mask.to(hidden_states.device)
            if cross_attn_layer_head_mask is not None:
                cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                raise NotImplementedError
        else:
            if i in rmt_parent.memory_storage:
                layer_memory = rmt_parent.memory_storage[i]
                non_empty_mask = rmt_parent.memory_storage['non_empty_mask']
                if layer_memory.shape[0] == 1:
                    layer_memory = layer_memory.repeat(len(non_empty_mask), 1, 1)
                
                hidden_states = torch.cat([layer_memory[non_empty_mask], hidden_states], dim=1)
                layer_attention_mask = extended_attention_mask
            else:
                layer_memory = None
                layer_attention_mask = extended_attention_mask[:, :, :, rmt_parent.num_mem_tokens:]
                
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=layer_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=layer_head_mask,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            if rmt_parent.memory_layers is not None:
                memory_layer = rmt_parent.memory_layers[i]
                memory_layer_out = memory_layer(hidden_states, 
                                                attention_mask=layer_attention_mask,
                                                position_bias=position_bias,
                                                encoder_hidden_states=encoder_hidden_states,
                                                encoder_attention_mask=encoder_extended_attention_mask,
                                                encoder_decoder_position_bias=encoder_decoder_position_bias,
                                                layer_head_mask=layer_head_mask,
                                                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                                                past_key_value=past_key_value,
                                                use_cache=use_cache,
                                                output_attentions=output_attentions,
                )

        # layer_outputs is a tuple with:
        # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
        if use_cache is False:
            layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

        hidden_states, present_key_value_state = layer_outputs[:2]
        
        #!! shorten hidden states
        if i in rmt_parent.memory_storage:
            hidden_states = hidden_states[:, memory_len:]
        
        if rmt_parent.memory_layers is not None:
            memory_layer_hidden_states = memory_layer_out[0]
            if i in rmt_parent.memory_storage:
                memory_layer_hidden_states = memory_layer_hidden_states[:, memory_len:]
            updated_memory = memory_layer_hidden_states[:, rmt_parent.memory_position]
            
            hidden_states[:, rmt_parent.memory_position] = updated_memory
        
        if layer_memory is not None:
            layer_memory[non_empty_mask] = hidden_states[:, rmt_parent.memory_position].detach()
        else:
            layer_memory = hidden_states[:, rmt_parent.memory_position].detach()
        rmt_parent.memory_storage[i] = layer_memory

        # We share the position biases between the layers - the first layer store them
        # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
        # (cross-attention position bias), (cross-attention weights)
        position_bias = layer_outputs[2]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
        # append next layer key value states
        if use_cache:
            present_key_value_states = present_key_value_states + (present_key_value_state,)

        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[3],)
            if self.is_decoder:
                all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

        # Model Parallel: If it's the last layer for that device, put things on the next device
        if self.model_parallel:
            for k, v in self.device_map.items():
                if i == v[-1] and "cuda:" + str(k) != self.last_device:
                    hidden_states = hidden_states.to("cuda:" + str(k + 1))

    hidden_states = self.final_layer_norm(hidden_states)
    hidden_states = self.dropout(hidden_states)

    # Add last layer
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v
            for v in [
                hidden_states,
                present_key_value_states,
                all_hidden_states,
                all_attentions,
                all_cross_attentions,
            ]
            if v is not None
        )
    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=present_key_value_states,
        hidden_states=all_hidden_states,
        attentions=all_attentions,
        cross_attentions=all_cross_attentions,
    )



def horizontal_memory_forward_1by1(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        rmt_parent=None,
    ):
    # Model parallel
    if self.model_parallel:
        torch.cuda.set_device(self.first_device)
        self.embed_tokens = self.embed_tokens.to(self.first_device)
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        err_msg_prefix = "decoder_" if self.is_decoder else ""
        raise ValueError(
            f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
        )
    elif input_ids is not None:
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        err_msg_prefix = "decoder_" if self.is_decoder else ""
        raise ValueError(f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds")

    if inputs_embeds is None:
        assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
        inputs_embeds = self.embed_tokens(input_ids)

    batch_size, seq_length = input_shape

    # required mask seq length can be calculated via length of past
    mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length
    # mask_seq_length += rmt_parent.num_mem_tokens
    
    # print('mask_seq_length', mask_seq_length)

    if use_cache is True:
        assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

    if attention_mask is None:
        attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        
    
    memory_len = rmt_parent.num_mem_tokens
    memory_attention_mask = torch.clone(attention_mask)[:, :memory_len]
    attention_mask = torch.cat([memory_attention_mask, attention_mask], dim=-1)

    
    if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
        encoder_seq_length = encoder_hidden_states.shape[1]
        encoder_attention_mask = torch.ones(
            batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
        )

    # initialize past_key_values with `None` if past does not exist
    if past_key_values is None:
        past_key_values = [None] * len(self.block)

    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

    # If a 2D or 3D attention mask is provided for the cross-attention
    # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
    if self.is_decoder and encoder_hidden_states is not None:
        encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
        encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
        if encoder_attention_mask is None:
            encoder_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
        encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
    else:
        encoder_extended_attention_mask = None

    # Prepare head mask if needed
    head_mask = self.get_head_mask(head_mask, self.config.num_layers)
    cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
    present_key_value_states = () if use_cache else None
    all_hidden_states = () if output_hidden_states else None
    all_attentions = () if output_attentions else None
    all_cross_attentions = () if (output_attentions and self.is_decoder) else None
    position_bias = None
    encoder_decoder_position_bias = None

    hidden_states = self.dropout(inputs_embeds)

    for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):
        layer_head_mask = head_mask[i]
        cross_attn_layer_head_mask = cross_attn_head_mask[i]
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(hidden_states.device)
            # Ensure that attention_mask is always on the same device as hidden_states
            if attention_mask is not None:
                attention_mask = attention_mask.to(hidden_states.device)
            if position_bias is not None:
                position_bias = position_bias.to(hidden_states.device)
            if encoder_hidden_states is not None:
                encoder_hidden_states = encoder_hidden_states.to(hidden_states.device)
            if encoder_extended_attention_mask is not None:
                encoder_extended_attention_mask = encoder_extended_attention_mask.to(hidden_states.device)
            if encoder_decoder_position_bias is not None:
                encoder_decoder_position_bias = encoder_decoder_position_bias.to(hidden_states.device)
            if layer_head_mask is not None:
                layer_head_mask = layer_head_mask.to(hidden_states.device)
            if cross_attn_layer_head_mask is not None:
                cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(hidden_states.device)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                raise NotImplementedError
        else:
            if i in rmt_parent.memory_storage:
                layer_memory = rmt_parent.memory_storage[i]
                non_empty_mask = rmt_parent.memory_storage['non_empty_mask']
                if layer_memory.shape[0] == 1:
                    layer_memory = layer_memory.repeat(len(non_empty_mask), 1, 1)
                
                hidden_states = torch.cat([layer_memory[non_empty_mask], hidden_states], dim=1)
                layer_attention_mask = extended_attention_mask
            else:
                layer_memory = None
                layer_attention_mask = extended_attention_mask[:, :, :, rmt_parent.num_mem_tokens:]
                
            layer_outputs = layer_module(
                hidden_states,
                attention_mask=layer_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                layer_head_mask=layer_head_mask,
                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            if rmt_parent.memory_layers is not None:
                memory_layer = rmt_parent.memory_layers[i]
                memory_layer_out = memory_layer(hidden_states, 
                                                attention_mask=layer_attention_mask,
                                                position_bias=position_bias,
                                                encoder_hidden_states=encoder_hidden_states,
                                                encoder_attention_mask=encoder_extended_attention_mask,
                                                encoder_decoder_position_bias=encoder_decoder_position_bias,
                                                layer_head_mask=layer_head_mask,
                                                cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                                                past_key_value=past_key_value,
                                                use_cache=use_cache,
                                                output_attentions=output_attentions,
                )

        # layer_outputs is a tuple with:
        # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
        if use_cache is False:
            layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

        hidden_states, present_key_value_state = layer_outputs[:2]
        
        #!! shorten hidden states
        if i in rmt_parent.memory_storage:
            hidden_states = hidden_states[:, memory_len:]
        
        if rmt_parent.memory_layers is not None:
            memory_layer_hidden_states = memory_layer_out[0]
            if i in rmt_parent.memory_storage:
                memory_layer_hidden_states = memory_layer_hidden_states[:, memory_len:]
            updated_memory = memory_layer_hidden_states[:, rmt_parent.memory_position]
            
            hidden_states[:, rmt_parent.memory_position] = updated_memory
        
        if layer_memory is not None:
            layer_memory[non_empty_mask] = hidden_states[:, rmt_parent.memory_position].detach()
        else:
            layer_memory = hidden_states[:, rmt_parent.memory_position].detach()
        rmt_parent.memory_storage[i] = layer_memory

        # We share the position biases between the layers - the first layer store them
        # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
        # (cross-attention position bias), (cross-attention weights)
        position_bias = layer_outputs[2]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
        # append next layer key value states
        if use_cache:
            present_key_value_states = present_key_value_states + (present_key_value_state,)

        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[3],)
            if self.is_decoder:
                all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

        # Model Parallel: If it's the last layer for that device, put things on the next device
        if self.model_parallel:
            for k, v in self.device_map.items():
                if i == v[-1] and "cuda:" + str(k) != self.last_device:
                    hidden_states = hidden_states.to("cuda:" + str(k + 1))

    hidden_states = self.final_layer_norm(hidden_states)
    hidden_states = self.dropout(hidden_states)

    # Add last layer
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    if not return_dict:
        return tuple(
            v
            for v in [
                hidden_states,
                present_key_value_states,
                all_hidden_states,
                all_attentions,
                all_cross_attentions,
            ]
            if v is not None
        )
    return BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=hidden_states,
        past_key_values=present_key_value_states,
        hidden_states=all_hidden_states,
        attentions=all_attentions,
        cross_attentions=all_cross_attentions,
    )