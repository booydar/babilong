from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from tqdm.auto import tqdm
from IPython.display import display

from babilong.metrics import compare_answers, TASK_LABELS

matplotlib.rc('font', size=14)
pd.set_option('display.max_colwidth', None)


TASKS = ['qa1', 'qa2', 'qa3', 'qa4', 'qa5']
LENGTHS = ['0k', '1k', '2k', '4k', '8k', '16k', '32k', '64k', '128k']


def get_model_results(results_path, tasks=TASKS, lengths=LENGTHS):
    run_configurations = set()
    for fn in results_path.glob('*.csv'):
        run_configurations.add('_'.join(fn.stem.split('_')[2:]))
    results = {}

    for run_cfg in run_configurations:
        accuracy = np.ones((len(tasks), len(lengths))) * -1
        for j, task in enumerate(tasks):
            for i, ctx_length in enumerate(lengths):
                fname = results_path / f'{task}_{ctx_length}_{run_cfg}'
                results_fname = fname.with_suffix('.csv')
                if not results_fname.exists():
                    continue

                df = pd.read_csv(results_fname)

                if df['output'].dtype != object:
                    df['output'] = df['output'].astype(str)
                df['output'] = df['output'].fillna('')

                df['correct'] = df.apply(lambda row: compare_answers(target=row['target'], output=row['output'],
                                                                     question=row['question'],
                                                                     task_labels=TASK_LABELS[task]), axis=1)
                score = df['correct'].sum()
                accuracy[j, i] = 100 * score / len(df) if len(df) > 0 else 0
                results[run_cfg] = accuracy
    return results


def parse_run_cfg(cfg_str):
    parts = cfg_str.split('_')
    result = {}
    key_parts = []
    for p in parts:
        if p in ("yes", "no"):
            key = "_".join(key_parts)
            result[key] = (p == "yes")
            key_parts = []
        else:
            key_parts.append(p)
    return result


def plot_results(model_name, results, lengths=LENGTHS, tasks=TASKS):
    # Create a colormap for the heatmap
    cmap = LinearSegmentedColormap.from_list('ryg', ["red", "yellow", "green"], N=256)

    # Create the heatmap
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 3.5))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    for i, run_cfg in enumerate(sorted(results.keys())):
        sns.heatmap(results[run_cfg], cmap=cmap, vmin=0, vmax=100, annot=True, fmt=".0f",
                    linewidths=.5, xticklabels=lengths, yticklabels=tasks, ax=axes[i])

        cfg_string = ''
        for k, v in parse_run_cfg(run_cfg).items():
            if v and k not in ['chat_template', 'system_prompt']:
                cfg_string += f'{k}+'
        cfg_string = cfg_string[:-1]

        # Set the main title.
        axes[i].set_title(f"{model_name.split('/')[-1]}\n")
        # Add a second line with a smaller font, positioned above the axis.
        axes[i].text(0.5, 1.05, cfg_string, transform=axes[i].transAxes, ha='center', fontsize=10)
        axes[i].set_xlabel('Context size')
        axes[i].set_ylabel('Tasks')
    return fig


def get_results_table(model_name, results, tasks=TASKS, lengths=LENGTHS, to_display=True):
    best_tab = None
    best_avgs = {}
    best_cfgs = {}  # Store best config for each task

    # Go through each configuration
    for cfg in results.keys():
        tab = results[cfg]
        tab = pd.DataFrame(tab, index=tasks, columns=lengths[:tab.shape[1]])
        tab['len_avg'] = tab.mean(axis=1)
        if to_display:
            print(f'{model_name}\n{cfg}')
            display(tab.iloc[:, :-1].round().astype(int))

        # For each task, check if this config gives better results
        for task in tasks:
            curr_avg = tab.loc[task, 'len_avg']
            if task not in best_avgs or curr_avg > best_avgs[task]:
                if best_tab is None:
                    best_tab = tab.copy()
                    best_tab['best_cfg'] = ''  # Add column for best configs
                best_tab.loc[task] = tab.loc[task]
                best_tab.loc[task, 'best_cfg'] = cfg
                best_avgs[task] = curr_avg
                best_cfgs[task] = cfg

    # Add average row
    best_tab.loc['avg'] = best_tab.iloc[:, :-2].mean(axis=0)
    best_tab.loc['avg', 'best_cfg'] = 'N/A'

    display_cols = list(best_tab.columns[:-2]) + ['best_cfg']
    if to_display:
        print(f'{model_name}\nbest setup:')
        display(best_tab[display_cols].round().astype({col: int for col in best_tab.columns[:-2]}))
        print(model_name)
        display(best_tab[display_cols[:-1]].round().astype({col: int for col in best_tab.columns[:-2]}))
    return best_tab


def process_single_model(model_name, evals_path, tasks, lengths, save_path=None):
    """Process results for a single model."""
    model_results = get_model_results(evals_path / model_name, tasks, lengths)
    table = get_results_table(model_name, model_results, tasks, lengths)

    if save_path:
        save_model_results(model_name, model_results, table, save_path)

    return model_results, table


def process_all_models(evals_path, tasks, lengths, save_path=None):
    """Process results for all models in the evals directory."""
    model_names = [f'{path.parent.name}/{path.name}' for path in evals_path.glob('*/*') if '.git' not in str(path)]
    print(f'Found predictions of following models:\n{model_names}')
    results = {}

    for model_name in tqdm(model_names, desc='Reading models predictions'):
        model_results = get_model_results(evals_path / model_name, tasks, lengths)
        table = get_results_table(model_name, model_results, tasks, lengths, to_display=False)
        results[model_name] = table

    for model_name in model_names:
        if save_path:
            save_model_results(model_name, model_results, table, save_path)

    if save_path and results:
        save_combined_results(results, save_path)

    return results


def save_model_results(model_name, model_results, table, save_path):
    """Save results for a single model to CSV and PDF files."""
    csv_path = save_path / f'{model_name}.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.iloc[:, :-2].to_csv(csv_path, index=True)

    fig = plot_results(model_name, model_results, args.lengths, args.tasks)
    fig.savefig(save_path / f'{model_name}.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'Results saved to {csv_path} and {save_path / f"{model_name}.pdf"}')


def save_combined_results(results, save_path):
    """Save combined results from all models to a single CSV file."""
    results_df = pd.concat([df.assign(model_name=model_name) for model_name, df in results.items()])
    results_df = results_df.reset_index()
    cols = [col for col in results_df.columns if col not in ['model_name', 'len_avg', 'best_cfg']]
    results_df[['model_name'] + cols].to_csv(save_path / 'all_results.csv', index=False)
    print(f'All results saved to {save_path / "all_results.csv"}')


# from babilong repo root directory:
# git lfs install
# download models predictions
# git clone https://huggingface.co/datasets/RMT-team/babilong_evals
# python -m babilong.collect_results --model_name all --save_path ./babilong_results --evals_path ./babilong_evals
# python -m babilong.collect_results --model_name meta-llama/Llama-3.2-1B-Instruct --save_path ./babilong_results --evals_path ./babilong_evals
# python -m babilong.collect_results --model_name meta-llama/Llama-3.2-1B-Instruct --tasks qa1 qa2 --lengths 0k 1k 2k --evals_path ./babilong_evals
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate results table for model performance')
    parser.add_argument('--evals_path', type=str, default='./babilong_evals',
                        help='Path to folder with models results.')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Model name, e.g., meta-llama/Llama-3.2-1B-Instruct')
    parser.add_argument('--tasks', nargs='+', default=TASKS, help='List of tasks, e.g. qa1 qa2 qa3 qa4 qa5')
    parser.add_argument('--lengths', nargs='+', default=LENGTHS, help='List of lengths, e.g. 0k 1k 2k')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save results')

    args = parser.parse_args()
    evals_path = Path(args.evals_path)
    save_path = Path(args.save_path) if args.save_path else None

    if args.model_name != 'all':
        process_single_model(args.model_name, evals_path, args.tasks, args.lengths, save_path)
    else:
        process_all_models(evals_path, args.tasks, args.lengths, save_path)
