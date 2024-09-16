import pandas as pd
import numpy as np
from preprocessing import preprocessing_data

file_name = 'whole/distance_record_whole_lg_weight_10.csv'
# Initialize the distance_metrics dictionary outside the loop
df = pd.read_csv(file_name, nrows=1)
if 'Unnamed: 0' in df.keys():
    df = df.drop('Unnamed: 0', axis=1)
total_cols = len(df.columns) - 2
chunk = 5000
## check if the total_cols can be divided by chunk
chunk_num = total_cols // chunk
right_times_chunk = []
distance_chunk = []
times_chunnk = []

for i in range(chunk_num):
    cols = ['subject_label', 'Inc_2']
    for j in range(chunk):
        cols.append('run_' + str(i*chunk+j))
    df = pd.read_csv(file_name, usecols=cols)
    label = df['Inc_2']
    ## get the value from 2 col to the end
    prediction = df.iloc[:, 2:]
    ## each column of prediction subtract label
    diff = prediction.sub(label, axis=0)
    ## get the absolute value of each element
    abs_diff = diff.abs()

    ## calculate the number of 0 for each row of abs_diff
    right_times = abs_diff.lt(0.5).sum(axis=1)
    distance = abs_diff.mean(axis=1)

    right_times_chunk.append(right_times)
    distance_chunk.append(distance)

if total_cols % chunk != 0:
    cols = ['subject_label', 'Inc_2']
    for j in range(chunk_num * chunk, total_cols, 1):
        cols.append('run_' + str(j))
    df = pd.read_csv(file_name, usecols=cols)
    label = df['Inc_2']
    ## get the value from 2 col to the end
    prediction = df.iloc[:, 2:]
    ## each column of prediction subtract label
    diff = prediction.sub(label, axis=0)
    ## get the absolute value of each element
    abs_diff = diff.abs()

    ## calculate the number of 0 for each row of abs_diff
    right_times = abs_diff.lt(0.5).sum(axis=1)
    distance = abs_diff.mean(axis=1)

    right_times_chunk.append(right_times)
    distance_chunk.append(distance)

right_times_sep = np.sum(right_times_chunk, axis=0)
distance_sep = np.nanmean(distance_chunk, axis=0)

distance_metrics = {'patient': df['subject_label'], 'label': df['Inc_2'], 'distance': distance_sep, 'right_times': right_times_sep}
df = pd.DataFrame(distance_metrics)
df = df.sort_values(by=['distance'], ascending=False)
df = df.reset_index(drop=True)
df.to_csv('whole/distance_metrics_lg_weight_10.csv')

## read the distance_metrics.csv
df = pd.read_csv('whole/distance_metrics_lg_weight_10.csv')
if 'Unnamed: 0' in df.keys():
    df = df.drop('Unnamed: 0', axis=1)
## change df patient to subject_label
df = df.rename(columns={'patient': 'subject_label'})

data = pd.read_csv(r'ClinicalDose_f.csv')
if 'Unnamed: 0' in data.keys():
    data = data.drop('Unnamed: 0', axis=1)

target = 'Inc_2'
fea_name = ['subject_label',
            'regcohortno',
            'demecog',
            'age_years',
            'small_bowel_v10',]

# data = preprocessing_data(data)
# fea_name = ['subject_label',
#             'regcohortno_Arm B (Standard + Irinotecan)',
#             'demecog_Fully Active',
#             'age_years',
#             'small_bowel_v15',]

features = data[fea_name]
## put the features into the distance_metrics so that subject_label and patient can be matched
distance_metrics = df.merge(features, on='subject_label')
distance_metrics.to_csv('whole/check2_lg_weight_10.csv')



