"""
    This python script is used to get rid of the header row of the mnist 
    csv file in order to make it easier to read into a C file
"""

import pandas as pd

#load csv file mnist_test.csv and mnist_train.csv without the header row
df_test = pd.read_csv('mnist_test.csv', header=None)
df_train = pd.read_csv('mnist_train.csv', header=None)

# remove row 1 (header row from the data)
df_test = df_test.drop(df_test.index[0])
df_train = df_train.drop(df_train.index[0])

#save the dataframes to csv files without the header row
df_test.to_csv('mnist_test.csv', header=False, index=False)
df_train.to_csv('mnist_train.csv', header=False, index=False)