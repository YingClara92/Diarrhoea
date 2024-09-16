import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import numpy as np
import os

def preprocessing_data(data, file_name):
    # data = pd.read_csv(r'check_whole_two_rm_target_10_d.csv')
    if 'Unnamed: 0' in data.keys():
        data = data.drop('Unnamed: 0', axis=1)
    id = data['subject_label']
    institute = data['institute']
    data = data.drop(['subject_label', 'institute'], axis=1)

    cat_cols = ['regcohortno', 'regsex', 'demecog']

    target = 'Inc_2'
    num_cols = list(set(data.keys()).difference(set(cat_cols)).difference(set([target])))
    ct = ColumnTransformer(
        [("CatTrans", OneHotEncoder(), cat_cols),
         ("NumTrans", MinMaxScaler(), num_cols), ],
        remainder='passthrough', verbose_feature_names_out=False)
    SubjectList = pd.DataFrame(ct.fit_transform(data), columns=ct.get_feature_names_out())

    clinical_cols = list(set(ct.get_feature_names_out()).difference(set([target])))
    np.save('ct.npy', ct)

    for col in clinical_cols:
        if 'nan' in col:
            clinical_cols.remove(col)

    clinical_cols.append('subject_label')
    clinical_cols.append('Inc_2')

    imp_mean = KNNImputer()

    imp_mean.fit(SubjectList)
    SubjectList = pd.DataFrame(imp_mean.transform(SubjectList), columns=SubjectList.columns)

    SubjectList['subject_label'] = id
    SubjectList['institute'] = institute
    data_csv = SubjectList

    distance_record = data_csv[['subject_label', 'Inc_2']]
    if file_name != '':
        ## check if the file exists
        if os.path.exists(file_name):
            distance_record = pd.read_csv(file_name)
            if 'Unnamed: 0' in distance_record.keys():
                distance_record = distance_record.drop('Unnamed: 0', axis=1)
        else:
            distance_record.to_csv(file_name)

    return data_csv[clinical_cols]