import pandas as pd
import numpy as np
import re
import nltk
from torch.utils.data import Dataset

# preprocess babi text files
def get_dataset_df(dataset_path, max_n_facts=None, max_n_samples=None):
    with open(dataset_path, 'r') as f:
        texts = f.read().strip()
        texts = texts.split('\n')
        df = pd.DataFrame(texts, columns=['text'])

    # parse samples
    df['phrase_num'] = df.text.apply(lambda x: int(x.split(' ')[0]))
    df.text = df.text.apply(lambda x: x[x.index(' ') + 1:])
    df['answer'] = df.text.apply(lambda x: x[x.index('\t') + 1:] if '\t' in x else None)
    # df['reference_num'] = df.answer.apply(lambda x: x if x is None else x.split('\t| ')[1:])
    df['reference_num'] = df.answer.apply(lambda x: x if x is None else [int(n) for n in re.split('\t| ', x)[1:]])
    df.answer = df.answer.apply(lambda x: x if x is None else x.split('\t')[0])
    df.text = df.text.apply(lambda x: x.split('\t')[0] if '\t' in x else x)

    # mark each sample
    sample_start_inds = list(np.where(df.phrase_num == 1)[0]) + [df.shape[0]]
    for i, (start, end) in enumerate(zip(sample_start_inds, sample_start_inds[1:])):
        df.loc[start:end, 'initial_sample_num'] = i

    df.initial_sample_num = df.initial_sample_num.astype(int)

    # multiple questions in sample -> samples with single question
    initial_samples = [df[df.initial_sample_num == sn] for sn in df.initial_sample_num.unique()]

    single_question_slices = []
    for sample in initial_samples:
        answer_positions = sample[~sample.answer.isna()].index
        slices = [sample.loc[:ans_pos].copy() for ans_pos in answer_positions]
        for i, slc in enumerate(slices):
            slices[i] = slc[(slc.answer.isna()) | (slc.index == slc.index[-1])]
        if max_n_facts is not None:             # drop samples with too many facts
            slices = [slc for slc in slices if slc.shape[0] <= max_n_facts]
        single_question_slices += slices
    
    if max_n_samples is not None:
        single_question_slices = single_question_slices[:max_n_samples]
    df = pd.concat(single_question_slices).reset_index(drop=True)

    # mark each sample again
    sample_start_inds = list(np.where(df.phrase_num == 1)[0]) + [df.shape[0]]
    for i, (start, end) in enumerate(zip(sample_start_inds, sample_start_inds[1:])):
        df.loc[start:end, 'sample_num'] = i

    df.sample_num = df.sample_num.astype(int)
    
    return df


# babi task loader dataset
class TaskDatasetRepeatReferences(Dataset):
    def __init__(self, dataset_path, max_n_facts=None, no_distractors=False):
        self.fact_dataset = get_dataset_df(dataset_path, max_n_facts=max_n_facts)
        self.no_distractors = no_distractors

    def __getitem__(self, ind):
        slc = self.fact_dataset[self.fact_dataset.sample_num == ind]
        references = slc[slc.phrase_num.isin(slc.reference_num.values[-1])].text.values
        facts = references if self.no_distractors else slc.text.values[:-1]
        sample = {'facts': facts,
                  'question': slc.text.values[-1],
                  'answer': slc.answer.values[-1],
                  'references': references}
        return sample

    def __len__(self):
        return self.fact_dataset.sample_num.max()