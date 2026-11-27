"""Summary tables (mean and sample SD), convergence plots and an optional Welch test."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from experiments.validation import validate_complete_suite


def analyse(csv_path, output, compare=None, metric='test_accuracy', paired=False):
    df = pd.read_csv(csv_path)
    validate_complete_suite(df, csv_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    columns = ['train_accuracy','test_accuracy','test_precision','test_recall','test_f1','seconds','evaluations']
    df.groupby('name')[columns].agg(['mean','std']).to_csv(output/'summary_numeric.csv')
    formatted = df.groupby('name')[columns].agg(lambda x: f'{x.mean():.6f} ± {x.std(ddof=1):.6f}')
    formatted.to_csv(output/'summary_mean_std.csv')
    for mode in ('accuracy','mse','bce'):
        fig, ax = plt.subplots(figsize=(10,6))
        count = 0
        for name, group in df.groupby('name'):
            config = json.loads(group.config_json.iloc[0])
            if config['fitness'] != mode:
                continue
            curves = np.array([json.loads(x) for x in group.history_json])
            evaluations = np.arange(1, curves.shape[1]+1)*config['swarm_size']
            mean, std = curves.mean(axis=0), curves.std(axis=0,ddof=1)
            ax.plot(evaluations,mean,label=name)
            ax.fill_between(evaluations,mean-std,mean+std,alpha=.12)
            count += 1
        if count:
            ax.set(xlabel='Fitness evaluations (initial swarm included)', ylabel=f'Best training fitness ({mode}; maximised)')
            ax.legend(fontsize=6,loc='upper left',bbox_to_anchor=(1,1))
            fig.savefig(output/f'convergence_{mode}.png',bbox_inches='tight',dpi=160)
        plt.close(fig)
    if compare:
        # Welch test by default; a paired test is only computed when --paired is given.
        a = df.loc[df.name == compare[0]].set_index('seed')[metric].sort_index()
        b = df.loc[df.name == compare[1]].set_index('seed')[metric].sort_index()
        if len(a)!=10 or len(b)!=10:
            raise ValueError('Both comparison names must exist')
        welch = stats.ttest_ind(a,b,equal_var=False)
        result = dict(a=compare[0],b=compare[1],metric=metric,
                      welch_t=welch.statistic,welch_p=welch.pvalue,
                      mean_difference=float(a.mean()-b.mean()))
        if paired:
            paired_result = stats.ttest_rel(a,b)
            result.update(paired_t=paired_result.statistic,paired_p=paired_result.pvalue)
        pd.DataFrame([result]).to_csv(output/'comparison.csv',index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--compare', nargs=2, metavar=('SETTING_A','SETTING_B'))
    parser.add_argument('--metric', default='test_accuracy', choices=['test_accuracy','test_f1','train_accuracy'])
    parser.add_argument('--paired', action='store_true', help='Also compute paired t-test; requires a justified paired design')
    args=parser.parse_args()
    analyse(args.csv,args.output,args.compare,args.metric,args.paired)
