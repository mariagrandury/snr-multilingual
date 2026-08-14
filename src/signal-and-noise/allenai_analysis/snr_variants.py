import warnings
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning) # supress function fitting warnings
pd.set_option('display.max_columns', None) # display all pandas cols

from snr.download.hf import pull_predictions_from_hf

local_path = pull_predictions_from_hf("allenai/signal-and-noise", split_name='datadecide_intermediate')
df = pd.read_parquet(local_path)
print(f'Loaded {len(df):,} model evaluations')

MIXES  = df['group'].unique()
SEEDS  = df['seed'].unique()
MODELS = df['model'].unique()
TASKS  = df['task'].unique()

SIZES = [
    '4M', '6M', '8M', '10M', '14M', '16M', '20M', 
    '60M', '90M', '150M', '300M', '530M', '750M', '1B'
]

# Default seed
SEED = 6198

# Default setup
selected_tasks = TASKS
metric = 'primary_metric'

###### SNR #######


from snr.dataloader import get_slice
import numpy as np
import pandas as pd
from itertools import product

metrics_list = []
for task, seed, mix, size in tqdm(product(selected_tasks, SEEDS, MIXES, SIZES), total=len(selected_tasks)*len(SEEDS)*len(MIXES)*len(SIZES)):
    curve_data = get_slice(df, mix=mix, task=task, size=size, seed=seed)
    if curve_data.empty: continue

    curve_data = curve_data.sort_values('step')
    curve_values = curve_data[metric].values

    # Calculate standard deviation and count of steps in last X% of checkpoints
    last_vals = curve_values[int(-0.3 * len(curve_values)):]
    step_std = np.std(last_vals)
    
    # Calculate final score (avg of last X)
    final_score = np.mean(curve_values[int(-0.1 * len(curve_values)):])
    score_last_1 = curve_values[-1]
    
    metrics_list += [{
        'task': task,
        'mix': mix,
        'size': size, 
        'seed': seed,
        'score': final_score,
        'score_last_n': final_score,
        'score_last_1': score_last_1,
        'step_std': step_std,
    }]

metrics_df = pd.DataFrame(metrics_list)

from itertools import product
from snr.stats import compute_decision_accuracy
from scipy.stats import pearsonr, spearmanr

# Calculate decision accuracy
mix_decision_accuracys = []
for seed, task, size in product(SEEDS, selected_tasks, SIZES):
    score_1b = metrics_df[(metrics_df['task'] == task) & (metrics_df['size'] == '1B')] # only one seed at the target scale
    score_size = metrics_df[(metrics_df['task'] == task) & (metrics_df['size'] == size) & (metrics_df['seed'] == seed)]

    if score_size.empty:
        continue

    # Get the mix orders and scores
    score_1b_sorted   = score_1b.sort_values('score', ascending=False).reset_index(drop=True)
    score_size_sorted = score_size.sort_values('score', ascending=False).reset_index(drop=True)
    mixes_1b          = score_1b_sorted['mix'].tolist()
    mixes_size        = score_size_sorted['mix'].tolist()
    scores_1b         = score_1b_sorted['score'].tolist()
    scores_size       = score_size_sorted['score'].tolist()
    
    decision_accuracy = compute_decision_accuracy(
        mixes_1b=mixes_1b, 
        mixes_size=mixes_size
    )

    # Calculate additional correlations
    common_mixes       = list(set(mixes_1b) & set(mixes_size))
    scores_1b_common   = [scores_1b[mixes_1b.index(mix)] for mix in common_mixes]
    scores_size_common = [scores_size[mixes_size.index(mix)] for mix in common_mixes]
    pearson_corr, _    = pearsonr(scores_1b_common, scores_size_common)
    spearman_corr, _   = spearmanr(scores_1b_common, scores_size_common)

    mix_decision_accuracys += [{
        'task': task,
        'size': size,
        'seed': seed,
        'mix_decision_accuracy': decision_accuracy,
        'mix_pearson': pearson_corr,
        'mix_spearman': spearman_corr,
    }]

# Add to metrics_df
correlation_df = pd.DataFrame(mix_decision_accuracys)
metrics_df = metrics_df.merge(correlation_df, on=['task', 'size', 'seed'], how='left', suffixes=('_y', ''))
metrics_df = metrics_df.drop([col for col in metrics_df.columns if col.endswith('_y')], axis=1)

# Calculate aggregated stats
agg_stats = metrics_df.groupby(['task', 'size', 'seed'])['score'].agg([
    'mean',
    'std'
]).reset_index()
metrics_df = metrics_df.merge(agg_stats, on=['task', 'size', 'seed'], how='left')
sizes = metrics_df['size'].unique()


from snr.snr_variants import AGGREGATION_FUNCTIONS
from matplotlib.pylab import LinAlgError

def get_r2_for_aggregation(x_vals, y_vals):
    """Get R^2 value for a given aggregation"""
    x_log = np.log10(x_vals)
    r = np.corrcoef(x_log, y_vals)[0,1]
    r2 = r**2
    return r2

# Calculate R^2 for each aggregation
results = []
for func_dict in AGGREGATION_FUNCTIONS:
    agg_func = func_dict['func']
    title = func_dict['title']
    latex = func_dict['latex']
    signal_xlabel = func_dict['signal_xlabel']

    snr_values = []
    decision_accuracies = []
    pearson_corrs = []
    spearman_corrs = []
    
    for size in SIZES[7:-1]:
        for task in metrics_df['task'].unique():
            if task == 'olmes_10_macro_avg':
                continue
                
            mix_data = metrics_df[(metrics_df['task'] == task) & (metrics_df['size'] == size)]

            signal, noise, snr = agg_func(
                mix_data['step_std'], # step-to-step std dev
                mix_data['score_last_1'].values, # data final ckpt score
                mix_data['std'], # data step-to-step std dev
                mix_data['score_last_n'], # data avg of last n scores
            )
            decision_accuracy = np.mean(mix_data['mix_decision_accuracy'])

            pearson_corr = np.mean(mix_data['mix_pearson'])
            spearman_corr = np.mean(mix_data['mix_spearman'])
            
            snr_values.append(snr)
            decision_accuracies.append(decision_accuracy)
            pearson_corrs.append(pearson_corr)
            spearman_corrs.append(spearman_corr)
    
    r2_dec_acc = get_r2_for_aggregation(snr_values, decision_accuracies)
    r2_pearson = get_r2_for_aggregation(snr_values, pearson_corrs)
    r2_spearman = get_r2_for_aggregation(snr_values, spearman_corrs)
    
    results.append((title, latex, r2_dec_acc, r2_pearson, r2_spearman))

results_df = pd.DataFrame(results, columns=['Measure of \\Signal{}', '', 'SNR vs. Decision Accuracy $R^2$', 'SNR vs. Pearson $R^2$', 'SNR vs. Spearman $R^2$'])
results_df = results_df.sort_values('SNR vs. Decision Accuracy $R^2$', ascending=False)
results_df = results_df.set_index('Measure of \\Signal{}')
results_df['SNR vs. Decision Accuracy $R^2$'] = results_df['SNR vs. Decision Accuracy $R^2$'].apply(lambda x: '{:.3f}'.format(x))
results_df['SNR vs. Pearson $R^2$'] = results_df['SNR vs. Pearson $R^2$'].apply(lambda x: '{:.3f}'.format(x))
results_df['SNR vs. Spearman $R^2$'] = results_df['SNR vs. Spearman $R^2$'].apply(lambda x: '{:.3f}'.format(x))


# print(results_df.style.to_latex(
#     column_format='lll',
#     position='h',
#     position_float='centering',
#     caption='SNR Variant Results',
#     label='tab:snr_variants'
# ))

print(results_df.to_markdown(index=True))


from matplotlib import ticker
import numpy as np
import matplotlib.pyplot as plt
from snr.plot import plot_snr_scatter, config_snr_ax
from snr.constants import PLOT_DIR

def plot_snr_variants(aggregation_functions, plot_only_snr=False):
    # Calculate R² values and sort aggregation methods
    r2_values = []
    for func_dict in aggregation_functions:
        agg_func = func_dict['func']

        snr_values = []
        decision_accuracies = []
        
        for size in SIZES[7:-1]:
            for task in metrics_df['task'].unique():
                if task == 'olmes_10_macro_avg':
                    continue
                    
                mix_data = metrics_df[(metrics_df['task'] == task) & (metrics_df['size'] == size)]
                
                _, _, snr = agg_func(
                    mix_data['step_std'],
                    mix_data['score_last_1'].values,
                    mix_data['std'],
                    mix_data['score_last_n']
                )
                decision_accuracies.append(np.mean(mix_data['mix_decision_accuracy']))
                snr_values.append(snr)
        
        try:
            r2 = get_r2_for_aggregation(snr_values, decision_accuracies)
        except LinAlgError:
            r2 = float('-inf')
        r2_values.append((func_dict, r2))
    
    r2_values.sort(key=lambda x: x[1], reverse=True)
    sorted_aggregation_functions = [func_dict for func_dict, r2 in r2_values]

    # Set up plot grid
    n_rows = len(sorted_aggregation_functions) if not plot_only_snr else (len(sorted_aggregation_functions)-1)//4 + 1
    n_cols = 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3.5*n_rows))
    axes = axes.reshape(n_rows, n_cols)

    task_names = {
        'arc_challenge': 'ARC-C', 'arc_easy': 'ARC-E', 'boolq': 'BoolQ',
        'csqa': 'CSQA', 'hellaswag': 'HS', 'openbookqa': 'OBQA',
        'piqa': 'PIQA', 'socialiqa': 'SocIQA', 'winogrande': 'WinoG',
        'mmlu': 'MMLU', 'olmes_10_macro_avg': 'OLMES-10'
    }

    # Initialize data storage
    all_data = {ax: {'x': [], 'y': [], 'texts': []} for row in axes for ax in row}

    # Collect and plot data
    for size in SIZES[7:-1]:
        size_data = {ax: {'x': [], 'y': [], 'tasks': []} for row in axes for ax in row}
        
        for task in metrics_df['task'].unique():
            if task == 'olmes_10_macro_avg':
                continue
                
            mix_data = metrics_df[(metrics_df['task'] == task) & (metrics_df['size'] == size)]
            decision_accuracy = np.mean(mix_data['mix_decision_accuracy'])
            
            for idx, func_dict in enumerate(sorted_aggregation_functions):
                agg_func = func_dict['func']
                
                row_idx = idx // 4 if plot_only_snr else idx
                col_idx = idx % 4 if plot_only_snr else 2
                
                if plot_only_snr and row_idx >= n_rows:
                    continue

                signal, noise, snr = agg_func(
                    mix_data['step_std'],
                    mix_data['score_last_1'].values,
                    mix_data['std'],
                    mix_data['score_last_n']
                )
                
                ax = axes[row_idx][col_idx]
                size_data[ax]['x'].append(snr)
                size_data[ax]['y'].append(decision_accuracy)
                size_data[ax]['tasks'].append(task)
                all_data[ax]['x'].append(snr)
                all_data[ax]['y'].append(decision_accuracy)

                if not plot_only_snr:
                    # Add signal and noise plots
                    for i, val in enumerate([signal, noise]):
                        ax = axes[row_idx][i]
                        size_data[ax]['x'].append(val)
                        size_data[ax]['y'].append(decision_accuracy) 
                        size_data[ax]['tasks'].append(task)
                        all_data[ax]['x'].append(val)
                        all_data[ax]['y'].append(decision_accuracy)

        # Create scatter plots
        for row in axes:
            for ax in row:
                data = size_data[ax]
                if data['x']:
                    texts = plot_snr_scatter(ax, data['x'], data['y'], data['tasks'], size, task_names)
                    all_data[ax]['texts'].extend(texts)

    # Configure axes
    for idx, func_dict in enumerate(sorted_aggregation_functions):
        title = func_dict['title']
        snr_xlabel = func_dict['snr_xlabel']

        row_idx = idx // 4 if plot_only_snr else idx
        if plot_only_snr and row_idx >= n_rows:
            continue
            
        if plot_only_snr:
            ax = axes[row_idx][idx % 4]
            config_snr_ax(ax, all_data[ax]['x'], all_data[ax]['y'],
                           all_data[ax]['texts'], '', True, False)
            ax.set_xscale('log')

            ax.set_title(title, fontsize=12)
            ax.set_xlabel(snr_xlabel, fontsize=12)
        else:
            for col_idx, ax in enumerate(axes[idx]):
                config_snr_ax(ax, all_data[ax]['x'], all_data[ax]['y'],
                               all_data[ax]['texts'], '', 
                               col_idx == 2, col_idx == 1)
                               
                
                if col_idx == 1:
                    ax.invert_xaxis()
                if col_idx == 2:
                    ax.set_xscale('log')

        # Format axis ticks and add legend to last plot
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.xaxis.set_minor_formatter(ticker.ScalarFormatter())
        
        is_last = (idx == len(sorted_aggregation_functions)-1)
        if is_last:
            ax.legend(ncols=2, title='Model Size',
                     title_fontproperties={'weight': 'bold', 'size': 9},
                     fontsize=9, loc='lower right')

    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/snr_variants{"_snr_only" if plot_only_snr else ""}.pdf', bbox_inches='tight')
    plt.show()

# plot_snr_variants(AGGREGATION_FUNCTIONS)
plot_snr_variants(AGGREGATION_FUNCTIONS, plot_only_snr=True)


