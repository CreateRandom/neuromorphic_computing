import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import matplotlib.patches as mpatches
import matplotlib.colors as colors

df = pd.read_csv('results/results.csv')

def infer_condition(row):

    if isinstance(row['activity_regularizer'],str):
        return row['activity_regularizer']

    if not np.isnan(row['dropout']):
        if row['dropout'] == 0.0:
            return 'default'
        return 'dropout'

    return 'dropout_scheduler'

df['condition'] = df.apply(infer_condition,axis=1)
# string rep back to array
df['accuracy'] = df['accuracy'].apply(lambda x: ast.literal_eval(x))
df['computation'] = df['computation'].apply(lambda x: ast.literal_eval(x))

df['best_acc'] = df['accuracy'].apply(lambda x: max(x))
df['latency_till_max'] = df['accuracy'].apply(lambda x: np.argmax(x))

def find_above_threshold(ary, threshold):
    args = np.argwhere(np.array(ary) > threshold).tolist()
    if args == []:
        return -1
    # return first case
    else:
        return int(args[0][0])

df['latency_till_96'] = df['accuracy'].apply(lambda x: find_above_threshold(x, threshold=0.96))

df['computation_till_96'] = df.apply(lambda x: x['computation'][x['latency_till_96']], axis=1)

df['computation_till_max'] = df.apply(lambda x: x['computation'][x['latency_till_max']], axis=1)

best_for_each_condition = df.groupby('condition').agg({'best_acc': 'max'})

# plot acc over time for each condition

color_dict = {'default': 'r', 'dropout' : 'b',
              'sparse': 'g', 'l2' : 'y', 'dropout_scheduler':'m'}

legend = []
for k, v in color_dict.items():
    patch = mpatches.Patch(color=v, label=k)
    legend.append(patch)
plt.figure(dpi=300)
indices = list(range(len(df.loc[0]['accuracy'])))
for i, row in df.iterrows():
    plt.plot(indices, row['accuracy'], color_dict[row['condition']])
plt.legend(handles= legend)
plt.title('Accuracy over time')
plt.xlabel('Timestep')
plt.ylabel('Accuracy')
plt.savefig('results/accuracy_over_time.png')

# plot metrics against one another
plt.figure(dpi=300)
colors = np.array([colors.to_rgb(color_dict[x]) for x in df['condition']])
plt.scatter(df['latency_till_max'], df['best_acc'], c=colors, s = df['computation_till_max'])
plt.title('Accuracy, latency and total amount of computation')
plt.xlabel('Latency')
plt.ylabel('Accuracy')
plt.legend(handles= legend)
plt.savefig('results/acc_latency_comp.png')


# best indices
def format_row(row):
    model_name = row['model_name'].replace('_', ' ')
    if row['best_acc'] > 0.96:
        return '{} & {} & {} & {} & {}\\\\'.format(row['condition'], model_name, round(row['best_acc'],3), round(row['computation_till_96'],3),
                                                row['latency_till_96'])
    else:

        return '{} & {} & {} & {} & {}\\\\'.format(row['condition'], model_name, round(row['best_acc'],3), '-',
                                                    '-')

def format_frame(df):
    lines = []
    lines.append('\hline')
    for i, row in df.iterrows():
        lines.append(format_row(row))
        lines.append('\hline')
    return lines

best = df.groupby('condition').idxmax()['best_acc']
best_conds = df.loc[best]

print('\n'.join(format_frame(best_conds)))
