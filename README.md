# Recurrent Memory Transformer implementation compatible with Hugging Face models


RMT is a memory-augmented segment-level recurrent Transformer. It achieves state-of-the art results on Hyperpartisan dataset and beats Transformer-XL on algorithmic tasks and LM with limited input and memory size.

>[paper](https://arxiv.org/abs/2304.11062) Scaling Transformer to 1M tokens and beyond with RMT

>[paper](https://arxiv.org/abs/2207.06881) [code](https://github.com/booydar/LM-RMT) Recurrent Memory Transformer

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/booydar/t5-experiments/blob/wip/notebooks/rmt_demo_lm.ipynb) Example: LM with RMT

Recurrent Memory Transformer is implemented as follows:

![**RMT**](img/RMT_scheme.png?raw=True)

We implement our memory mechanism with no changes to Transformer model by adding special memory tokens to the input sequence. The model is trained to control both memory operations and sequence representations processing.

## Installation
```bash
pip install -e .
```
This command will install `lm_experiments_tools` with only required packages for Trainer and tools.

`lm_experiments_tools` Trainer supports gradient accumulation, logging to tensorboard, saving the best models
based on metrics, custom metrics and data transformations support.

### Install requirements for all experiments
Full requirements for all experiments are specified in requirements.txt. Install requirements after cloning the repo:
```bash
pip install -r requirements.txt
```


## Citation
If you find our work useful, please cite the RMT papers:
```
@inproceedings{
        bulatov2022recurrent,
        title={Recurrent Memory Transformer},
        author={Aydar Bulatov and Yuri Kuratov and Mikhail Burtsev},
        booktitle={Advances in Neural Information Processing Systems},
        editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
        year={2022},
        url={https://openreview.net/forum?id=Uynr3iPhksa}
}
```
```
@misc{bulatov2023scaling,
      title={Scaling Transformer to 1M tokens and beyond with RMT}, 
      author={Aydar Bulatov and Yuri Kuratov and Mikhail S. Burtsev},
      year={2023},
      eprint={2304.11062},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```