# class AssociativeRecurrentWrapper(torch.nn.Module, PyTorchModelHubMixin):
# ...
# from modeling_amt.language_modeling import *

HF_PATH = "booydar/RMT-Llama-3.2-1B-Instruct-4x1024-mem16-lora-babilong-qa1-5"

from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType

model = AutoModelForCausalLM.from_pretrained("/home/jovyan/kuratov/models/Llama-3.2-1B")

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, 
    inference_mode=False, 
    r=8, 
    lora_alpha=32, 
    lora_dropout=0.1
    )
model = get_peft_model(model, peft_config)

mem_cell_args = dict(
    base_model=model.cpu(),
    num_mem_tokens=16,
)
mem_cell_args['d_mem'] = 64
mem_cell_args['wrap_pos'] = False
mem_cell_args['correction'] = False
mem_cell_args['layers_attr'] = "base_model.base_model.layers"

cell = AssociativeMemoryCell(**mem_cell_args)
model = AssociativeRecurrentWrapper.from_pretrained(HF_PATH,
                                                  memory_cell=cell, 
                                                    segment_size=1024,
                                                    max_n_segments=4, 
                                                    vary_n_segments=False,
                                                    k2=-1,
                                                    return_all_logits=False,)