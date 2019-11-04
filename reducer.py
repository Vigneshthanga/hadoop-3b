#!/usr/bin/env python
"""reducer.py"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import sys

dframe = pd.read_csv(sys.stdin)

col = dframe.head(1)

dframe_maj = dframe[dframe.income == 0]
dframe_min = dframe[dframe.income == 1]

n_min_sample = len(dframe_min)


dframe_majority_downsampled = resample(dframe_maj,
                                 replace=True,     # sample with replacement
                                 n_samples=n_min_sample,    # to match minority class
                                 random_state=200) # reproducible results

# Combine minority class with downsampled majority class
dframe = pd.concat([dframe_min, dframe_majority_downsampled])

labels = dframe['income']

feature_set = dframe.iloc[:,0:14]

feature_set = pd.get_dummies(feature_set)

X_train, X_test, y_train, y_test = train_test_split(feature_set, labels, test_size=0.3, random_state=1)


one_hot_encoded_training_predictors = pd.get_dummies(X_train)
one_hot_encoded_test_predictors = pd.get_dummies(X_test)
final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
                                                                    join='left',
                                                                    axis=1)
print(final_train)
print(final_test)
print(y_train)
print(y_test)
