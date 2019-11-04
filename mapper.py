import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
dframe = pd.read_csv(sys.stdin,na_values="?")

col = dframe.head(1)

#Get Column list
col_list = list(col)

for c in col_list:
    indexNames = dframe[ dframe[c] == '?'].index
    dframe.drop(indexNames , inplace=True)

dframe = dframe.dropna()
col_list = list(col)

dframe['income'].astype(str)
dframe['income'] = dframe['income'] == '>50K'

dframe['income'] = dframe['income'].astype(int)

#print(dframe)
dframe.to_csv('inp.csv')
