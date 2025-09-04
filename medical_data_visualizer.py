import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv', sep=',')
print(df.columns)

# 2
bmi = df['weight'] / ((df['height'] / 100) ** 2)
df['bmi'] = bmi

df['overweight'] = (df['bmi'] > 25).astype(int)

# 3

df.loc[df['cholesterol'] == 1, 'cholesterol'] = 0
df.loc[df['cholesterol'] > 1, 'cholesterol'] = 1

df.loc[df['gluc'] == 1 , 'gluc'] = 0
df.loc[df['gluc'] > 1, 'gluc' ] = 1

# 4
def draw_cat_plot():
    
    # 5
    df_cat = pd.melt(df, id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])


    # 6
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')
    

    # 7
    catplot = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar')


    # 8
    fig = catplot.figure 


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ].copy()

    # Remover a coluna 'bmi' se existir
    if 'bmi' in df_heat.columns:
        df_heat = df_heat.drop(columns=['bmi'])


    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))



    # 14
    fig, ax = plt.subplots(figsize=(12, 9))

    # 15
    sns.heatmap(corr, mask=mask, annot=True, fmt='.1f', center=0, square=True, ax=ax, cbar_kws={'shrink': 0.5})


    # 16
    fig.savefig('heatmap.png')
    return fig
