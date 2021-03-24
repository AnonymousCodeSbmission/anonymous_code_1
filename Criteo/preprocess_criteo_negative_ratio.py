import os
import numpy as np
import pandas as pd
import random

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I' + str(i) for i in range(1, 14)]

data_directory = 'data/'
csv_file_name = os.path.join(data_directory, 'train_1.csv')
full_data = pd.read_csv(csv_file_name)

# throw away some negative examples to increase the ratio of the pos_ratio
new_neg_ratio = 0.5 # the target fraction of negative examples in the trimmed dataset

pos_ratio = np.sum(full_data['Label']) / full_data.shape[0]
print("pos_ratio: {}".format(pos_ratio))

neg_keep_ratio = pos_ratio * new_neg_ratio / (1.0 - new_neg_ratio) / (1.0 - pos_ratio)

row_choices = [] # indicator of which rows in the original data frame to keep
for num in full_data['Label']:
    if num == 1 or random.random() < neg_keep_ratio:
        row_choices.append(True)
    else:
        row_choices.append(False)
full_data = full_data.loc[row_choices]

new_pos_ratio = np.sum(full_data['Label']) / full_data.shape[0]
print("new_pos_ratio: {}".format(new_pos_ratio))


# fill in NA's
full_data[sparse_features] = full_data[sparse_features].fillna('',)
full_data[dense_features] = full_data[dense_features].fillna(0,)

# label encoding for categorical features
label_encoder_dict = {}
for feat in sparse_features:
    lbe = LabelEncoder() # encode target labels with value between 0 and n_classes-1.
    full_data.loc[:,feat] = lbe.fit_transform(full_data[feat]) # fit label encoder and return encoded label
    full_data.loc[:,feat] = full_data[feat].astype(np.int32) # convert from float64 to float32
    label_encoder_dict[feat] = lbe # store the fitted label encoder

# do simple Transformation for dense features
mms = MinMaxScaler(feature_range=(0, 1))
full_data.loc[:,dense_features] = mms.fit_transform(full_data[dense_features])
full_data.loc[:,dense_features] = full_data[dense_features].astype(np.float32)

for key in dense_features:
    print(key)
    print(np.max(full_data[key]), np.min(full_data[key]))

train_data, test_data = train_test_split(full_data, test_size=0.1, random_state=42)

for key in dense_features:
    print('train')
    print(np.max(train_data[key]), np.min(train_data[key]))
    print('test')
    print(np.max(test_data[key]), np.min(test_data[key]))

for key in sparse_features:
    print(key)
    print('train', train_data[key].nunique())
    print('test', test_data[key].nunique())

train_data.to_csv(path_or_buf='data/new_train_negfrac{}_0.9.csv'.format(new_neg_ratio), index=False)
test_data.to_csv(path_or_buf='data/new_test_negfrac{}_0.1.csv'.format(new_neg_ratio), index=False)