import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import matplotlib.patches as mpatches
import matplotlib.colors as colors

df = pd.read_csv('results.csv')

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

df['computation_till_max'] = df.apply(lambda x: x['computation'][x['latency_till_max']], axis=1)

best_for_each_condition = df.groupby('condition').agg({'best_acc': 'max'})

# plot acc over time for each condition

color_dict = {'default': 'r', 'dropout' : 'b',
              'sparse': 'g', 'l2' : 'y', 'dropout_scheduler':'m'}

legend = []
for k, v in color_dict.items():
    patch = mpatches.Patch(color=v, label=k)
    legend.append(patch)
plt.figure()
indices = list(range(len(df.loc[0]['accuracy'])))
for i, row in df.iterrows():
    plt.plot(indices, row['accuracy'], color_dict[row['condition']])
plt.legend(handles= legend)
plt.title('Accuracy over time')
plt.xlabel('Timestep')
plt.ylabel('Accuracy')
plt.show()

# plot metrics against one another
plt.figure()
colors = np.array([colors.to_rgb(color_dict[x]) for x in df['condition']])
plt.scatter(df['latency_till_max'], df['best_acc'], c=colors, s = df['computation_till_max'])
plt.title('Accuracy, latency and total amount of computation')
plt.xlabel('Latency')
plt.ylabel('Accuracy')
plt.legend(handles= legend)
plt.show()

