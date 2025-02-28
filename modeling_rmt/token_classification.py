import math
import torch
import torch.nn.functional as F
from .base import RMTBaseModel


class RMTEncoderForTokenClassification(RMTBaseModel):
    # todo: move segment looping into RMT class, also move help functions into RMT class
    def __init__(self, base_model, **rmt_kwargs):
        super().__init__(base_model, **rmt_kwargs)
        self.rmt_config['sum_loss'] = True

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, labels_mask=None, pos_weight=None, output_attentions=None,
                output_hidden_states=None, return_dict=None):
        # todo: currently output from RMT model is not the same like from backbone model with 1 segment
        # because of inserted memory tokens and operations with cls/sep/pad in pad_and_segment
        # need to impl such that output from forward is like output from backbone model:
        # input -> segmented_inp -> segmented_logits -> output
        #                               | -> loss         | -> metrics
        #                           segmented_labels <- labels

        kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids,
                  'position_ids': position_ids, 'head_mask': head_mask, 'inputs_embeds': inputs_embeds,
                  'labels': labels, 'labels_mask': labels_mask, 'pos_weight': pos_weight,
                  'output_attentions': output_attentions, 'output_hidden_states': output_hidden_states,
                  'return_dict': return_dict,
                  }
        memory = self.set_memory(input_ids.shape)
        segmented = self.pad_and_segment(input_ids, labels, labels_mask)

        base_model_outputs = []
        for seg_num, segment in enumerate(zip(*segmented)):
            if self.rmt_config['bptt_depth'] > -1:
                raise NotImplementedError

            seg_kwargs, non_empty_mask = self.prepare_kwargs(segment, kwargs)
            if sum(non_empty_mask) == 0:
                continue
            seg_kwargs['inputs_embeds'][:, self.memory_position] = memory[non_empty_mask]

            out = self.model(**seg_kwargs)
            memory[non_empty_mask] = out.hidden_states[-1][:, self.memory_position]

            out['seg_kwargs'] = seg_kwargs
            base_model_outputs.append(out)

        out = self.process_outputs(input_ids, base_model_outputs, output_attentions, output_hidden_states)

        return out

    def prepare_kwargs(self, segment, kwargs):
        segment_input_ids, segment_labels, segment_labels_mask = segment
        seg_kwargs = dict(**kwargs)
        non_empty_mask = [s is not None for s in segment_input_ids]
        if sum(non_empty_mask) == 0:
            return None, non_empty_mask

        input_ids = torch.stack([s for s in segment_input_ids if s is not None])
        inputs_embeds = self.model.embeddings(input_ids)

        seg_kwargs['input_ids'] = None
        seg_kwargs['inputs_embeds'] = inputs_embeds
        seg_kwargs['attention_mask'] = self.get_attention_mask(input_ids)
        if seg_kwargs.get('token_type_ids') is not None:
            seg_kwargs['token_type_ids'] = self.get_token_type_ids(input_ids)
        seg_kwargs['output_hidden_states'] = True
        if seg_kwargs['labels'] is not None:
            seg_kwargs['labels'] = torch.stack([el for el, m in zip(segment_labels, non_empty_mask) if m])
        if seg_kwargs['labels_mask'] is not None:
            seg_kwargs['labels_mask'] = torch.stack([el for el, m in zip(segment_labels_mask, non_empty_mask) if m])
        if kwargs['pos_weight'] is not None:
            pos_weight = kwargs['pos_weight']
            # all values in the second dimension of pos_weight should be the same
            pos_weight = pos_weight[0, 0, :][None, None, :]
            segm_bs, segm_seq_len, _ = seg_kwargs['labels'].shape
            seg_kwargs['pos_weight'] = pos_weight.repeat(segm_bs, segm_seq_len, 1)

        return seg_kwargs, non_empty_mask

    def process_outputs(self, input_ids, model_outputs, output_attentions, output_hidden_states):
        rmt_out = model_outputs[-1]

        bs, seq_len = input_ids.shape

        losses = []
        logits = []
        logits_masks = []
        labels_segm = []
        for out in model_outputs:
            losses.append(out['loss'])
            logits.append(out['logits'].detach())
            labels_segm += [out['seg_kwargs']['labels']]

            if out['seg_kwargs']['labels_mask'] is not None:
                logits_masks.append(out['seg_kwargs']['labels_mask'])

        # drop unnecessary hiddens to save memory
        if not output_hidden_states:
            for key in rmt_out.keys():
                if 'hidden_state' in key:
                    rmt_out[key] = None

        for i, l in enumerate(losses):
            rmt_out[f'loss_{i}'] = l.mean()

        # aggregate losses from all segments
        rmt_out['loss'] = torch.stack(losses).mean()

        # some sequences are skipped in some batches if they are empty, we need to put dummy predictions for them.
        # this may lead to different order of samples in the batch, but we modify order of labels and masks as well
        for i in range(len(logits)):
            logits[i] = F.pad(logits[i], (0, 0, 0, 0, 0, bs - logits[i].shape[0]))
            labels_segm[i] = F.pad(labels_segm[i], (0, 0, 0, 0, 0, bs - labels_segm[i].shape[0]))
            if len(logits_masks) > 0:
                logits_masks[i] = F.pad(logits_masks[i], (0, 0, 0, bs - logits_masks[i].shape[0]))

        rmt_out['logits'] = torch.cat(logits, dim=1)
        # Warning: rmt logits, labels, masks are not in the same order as in input data:
        # the first dimension is number of segments!
        # so, torch.cat will result in segm0, segm0,.. and only after all segm0 will come segm1, ... .
        # not segm0, segm1, segm0, segm1 as in input data
        rmt_out['logits_segm'] = [logits]
        rmt_out['labels_segm'] = [labels_segm]
        if len(logits_masks) > 0:
            rmt_out['rmt_logits_masks'] = torch.cat(logits_masks, dim=1)
            rmt_out['rmt_logits_masks_segm'] = [logits_masks]

        return rmt_out

    def pad_and_segment(self, input_ids, labels=None, labels_mask=None):
        segmented_batch = []
        segmented_batch_labels = []
        segmented_batch_labels_mask = []

        if labels is None:
            labels = [None] * input_ids.shape[0]
        batch_labels = labels

        if labels_mask is None:
            labels_mask = [None] * input_ids.shape[0]
        batch_labels_mask = labels_mask

        for seq, labels, labels_mask in zip(input_ids, batch_labels, batch_labels_mask):
            content_tokens_mask = (seq != self.pad_token_id) & (seq != self.cls_token.item()) & (seq != self.sep_token.item())
            seq = seq[content_tokens_mask]
            seq = seq[:self.segment_size * self.rmt_config['max_n_segments']]
            if labels is not None:
                labels = labels[content_tokens_mask]
                labels = labels[:self.segment_size * self.rmt_config['max_n_segments']]
            if labels_mask is not None:
                labels_mask = labels_mask[content_tokens_mask]
                labels_mask = labels_mask[:self.segment_size * self.rmt_config['max_n_segments']]

            # n_seg = math.ceil(len(seq) / self.segment_size)
            # input_segments = torch.chunk(seq, n_seg)
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

            if labels is not None:
                labels_segments = [labels[start:end] for (start, end) in zip(split_inds, split_inds[1:])]
                labels_segments = [self.pad_add_special_tokens(t, self.rmt_config['input_size'], add_to='labels') for t in labels_segments]
                labels_segments = [None] * n_empty_segments + labels_segments
                segmented_batch_labels.append(labels_segments)

            if labels_mask is not None:
                labels_mask_segments = [labels_mask[start:end] for (start, end) in zip(split_inds, split_inds[1:])]
                labels_mask_segments = [self.pad_add_special_tokens(t, self.rmt_config['input_size'], add_to='labels_mask') for t in labels_mask_segments]
                labels_mask_segments = [None] * n_empty_segments + labels_mask_segments
                segmented_batch_labels_mask.append(labels_mask_segments)

        segmented_batch = [[sample[seg_num] for sample in segmented_batch]
                           for seg_num in range(self.rmt_config['max_n_segments'])]
        segmented_batch_labels = [[sample[seg_num] for sample in segmented_batch_labels]
                                  for seg_num in range(self.rmt_config['max_n_segments'])]
        segmented_batch_labels_mask = [[sample[seg_num] for sample in segmented_batch_labels_mask]
                                       for seg_num in range(self.rmt_config['max_n_segments'])]

        return segmented_batch, segmented_batch_labels, segmented_batch_labels_mask

    def pad_add_special_tokens(self, tensor, segment_size, add_to='inputs'):
        input_elements = []
        if add_to == 'inputs':
            input_elements += [self.cls_token, self.mem_token_ids, self.sep_token, tensor, self.sep_token]
        elif add_to == 'labels':
            masked_labels = torch.zeros((1, tensor.shape[-1]), device=tensor.device)
            input_elements += [masked_labels, masked_labels.repeat(self.num_mem_tokens, 1), masked_labels, tensor, masked_labels]
        elif add_to == 'labels_mask':
            mask_value = torch.zeros((1), device=tensor.device)
            input_elements += [mask_value, mask_value.repeat(self.num_mem_tokens), mask_value, tensor, mask_value]

        tensor = torch.cat(input_elements)

        pad_size = segment_size - tensor.shape[0]
        if pad_size > 0:
            if add_to == 'inputs':
                tensor = F.pad(tensor, (0, pad_size), value=self.pad_token_id)
            elif add_to == 'labels':
                # todo: labels pad value should be specified, if not multilable classification it could be just -100
                tensor = F.pad(tensor, (0, 0, 0, pad_size), value=0)
            elif add_to == 'labels_mask':
                tensor = F.pad(tensor, (0, pad_size), value=0)
        return tensor


class RMTEncoderForMaskedLM(RMTBaseModel):
    def __init__(self, base_model, **rmt_kwargs):
        super().__init__(base_model, **rmt_kwargs)
        self.rmt_config['sum_loss'] = True

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, labels=None,
                output_attentions=None, output_hidden_states=None, return_dict=None,):
        # todo: currently output from RMT model is not the same like from backbone model with 1 segment
        # because of inserted memory tokens and operations with cls/sep/pad in pad_and_segment
        # need to impl such that output from forward is like output from backbone model:
        # input -> segmented_inp -> segmented_logits -> output
        #                               | -> loss         | -> metrics
        #                           segmented_labels <- labels
        kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids,
                  'position_ids': position_ids, 'head_mask': head_mask, 'inputs_embeds': inputs_embeds,
                  'encoder_hidden_states': encoder_hidden_states, 'encoder_attention_mask': encoder_attention_mask,
                  'labels': labels, 'output_attentions': output_attentions,
                  'output_hidden_states': output_hidden_states, 'return_dict': return_dict
                  }
        memory = self.set_memory(input_ids.shape)
        segmented = self.pad_and_segment(input_ids, labels)

        base_model_outputs = []
        for seg_num, segment in enumerate(zip(*segmented)):
            if self.rmt_config['bptt_depth'] > -1:
                raise NotImplementedError

            seg_kwargs, non_empty_mask = self.prepare_kwargs(segment, kwargs)
            if sum(non_empty_mask) == 0:
                continue
            seg_kwargs['inputs_embeds'][:, self.memory_position] = memory[non_empty_mask]

            out = self.model(**seg_kwargs)
            memory[non_empty_mask] = out.hidden_states[-1][:, self.memory_position]

            out['seg_kwargs'] = seg_kwargs
            base_model_outputs.append(out)

        out = self.process_outputs(input_ids, base_model_outputs, output_attentions, output_hidden_states)

        return out

    def prepare_kwargs(self, segment, kwargs):
        segment_input_ids, segment_labels = segment
        seg_kwargs = dict(**kwargs)
        non_empty_mask = [s is not None for s in segment_input_ids]
        if sum(non_empty_mask) == 0:
            return None, non_empty_mask

        input_ids = torch.stack([s for s in segment_input_ids if s is not None])
        inputs_embeds = self.model.embeddings(input_ids)

        seg_kwargs['input_ids'] = None
        seg_kwargs['inputs_embeds'] = inputs_embeds
        seg_kwargs['attention_mask'] = self.get_attention_mask(input_ids)
        if seg_kwargs.get('token_type_ids') is not None:
            seg_kwargs['token_type_ids'] = self.get_token_type_ids(input_ids)
        seg_kwargs['output_hidden_states'] = True
        if seg_kwargs['labels'] is not None:
            seg_kwargs['labels'] = torch.stack([el for el, m in zip(segment_labels, non_empty_mask) if m])

        return seg_kwargs, non_empty_mask

    def process_outputs(self, input_ids, model_outputs, output_attentions, output_hidden_states):
        rmt_out = model_outputs[-1]

        bs, seq_len = input_ids.shape

        losses = []
        logits = []
        labels_segm = []
        for out in model_outputs:
            losses.append(out['loss'])
            logits.append(out['logits'].detach())
            labels_segm += [out['seg_kwargs']['labels']]

        # drop unnecessary hiddens to save memory
        if not output_hidden_states:
            for key in rmt_out.keys():
                if 'hidden_state' in key:
                    rmt_out[key] = None

        for i, l in enumerate(losses):
            rmt_out[f'loss_{i}'] = l.mean()

        # aggregate losses from all segments
        rmt_out['loss'] = torch.stack(losses).mean()

        # some sequences are skipped in some batches if they are empty, we need to put dummy predictions for them.
        # this may lead to different order of samples in the batch, but we modify order of labels and masks as well
        for i in range(len(logits)):
            logits[i] = F.pad(logits[i], (0, 0, 0, 0, 0, bs - logits[i].shape[0]))
            labels_segm[i] = F.pad(labels_segm[i], (0, 0, 0, bs - labels_segm[i].shape[0]), value=-100)

        rmt_out['logits'] = torch.cat(logits, dim=1)
        # Warning: rmt logits, labels, masks are not in the same order as in input data:
        # the first dimension is number of segments!
        # so, torch.cat will result in segm0, segm0,.. and only after all segm0 will come segm1, ... .
        # not segm0, segm1, segm0, segm1 as in input data
        rmt_out['logits_segm'] = [logits]
        rmt_out['labels_segm'] = [labels_segm]

        return rmt_out

    def pad_and_segment(self, input_ids, labels=None):
        segmented_batch = []
        segmented_batch_labels = []

        if labels is None:
            labels = [None] * input_ids.shape[0]
        batch_labels = labels

        for seq, labels in zip(input_ids, batch_labels):
            content_tokens_mask = (seq != self.pad_token_id) & (seq != self.cls_token.item()) & (seq != self.sep_token.item())
            seq = seq[content_tokens_mask]
            seq = seq[:self.segment_size * self.rmt_config['max_n_segments']]
            if labels is not None:
                labels = labels[content_tokens_mask]
                labels = labels[:self.segment_size * self.rmt_config['max_n_segments']]

            # n_seg = math.ceil(len(seq) / self.segment_size)
            # input_segments = torch.chunk(seq, n_seg)
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

            if labels is not None:
                labels_segments = [labels[start:end] for (start, end) in zip(split_inds, split_inds[1:])]
                labels_segments = [self.pad_add_special_tokens(t, self.rmt_config['input_size'], add_to='labels') for t in labels_segments]
                labels_segments = [None] * n_empty_segments + labels_segments
                segmented_batch_labels.append(labels_segments)

        segmented_batch = [[sample[seg_num] for sample in segmented_batch]
                           for seg_num in range(self.rmt_config['max_n_segments'])]
        segmented_batch_labels = [[sample[seg_num] for sample in segmented_batch_labels]
                                  for seg_num in range(self.rmt_config['max_n_segments'])]

        return segmented_batch, segmented_batch_labels

    def pad_add_special_tokens(self, tensor, segment_size, add_to='inputs'):
        input_elements = []
        pad_value = 0
        if add_to == 'inputs':
            pad_value = self.pad_token_id
            input_elements += [self.cls_token, self.mem_token_ids, self.sep_token, tensor, self.sep_token]
        elif add_to == 'labels':
            pad_value = -100
            masked_labels = torch.ones((1), device=tensor.device, dtype=tensor.dtype) * pad_value
            input_elements += [masked_labels, masked_labels.repeat(self.num_mem_tokens), masked_labels, tensor, masked_labels]

        tensor = torch.cat(input_elements)

        pad_size = segment_size - tensor.shape[0]
        if pad_size > 0:
            tensor = F.pad(tensor, (0, pad_size), value=pad_value)
        return tensor
