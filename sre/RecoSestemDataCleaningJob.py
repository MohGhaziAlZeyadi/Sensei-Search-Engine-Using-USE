import os
import sys
import csv, jsonlines
import numpy as np
import copy
import random
import pandas as pd


df = pd.read_csv("C:/Users/Moh/SemanticSearchwithApproximateNearestNeighborsandTextEmbeddings/data/Reco/EventforReco.csv")
print(df.shape)

df = df.astype(str)

#convert to int ID
df['DelarID'] = df.groupby(['Delar']).ngroup()
max_DelarID = df['DelarID'].max()
print("max_DelarID :", max_DelarID)

df['DocID'] = df.groupby(['Doc']).ngroup()
max_DocID = df['DocID'].max()
print("max_DocID :", max_DocID)
#DelarID	DocID

Total_Number_Of_Delars = len(df['DelarID'].unique())
print("Total Number of delars are :", Total_Number_Of_Delars)

Total_Number_Of_Doc = len(df['DocID'].unique())
print("Total Number of doc are :", Total_Number_Of_Doc)

df =  df.dropna()
print("df after dropna()",df.shape)
df.insert(2, 'count', 1)
print("df after add count colme", df.shape)


#df = df.groupby(['DelarID', 'DocID']).sum()[["count"]]

new_df = df.groupby(['DelarID', 'DocID'])['count'].sum().reset_index()

print("df after aggregate and count",new_df.shape)
print(new_df.head)

from sklearn.utils import shuffle
new_df = shuffle(new_df)


Total_Number_Of_Delars = len(new_df['DelarID'].unique())
print("Total Number of delars are :", Total_Number_Of_Delars)

Total_Number_Of_Doc = len(new_df['DocID'].unique())
print("Total Number of doc are :", Total_Number_Of_Doc)
new_df.to_csv(r'C:/Users/Moh/SemanticSearchwithApproximateNearestNeighborsandTextEmbeddings/data/Reco/ua.base', index = False, header= False)



# Randomly sample 70% of your dataframe
ua_test = new_df.sample(frac=0.1)
ua_test.to_csv(r'C:/Users/Moh/SemanticSearchwithApproximateNearestNeighborsandTextEmbeddings/data/Reco/ua.test', index = False, header= False)


ub_test = new_df.sample(frac=0.1)
ub_test.to_csv(r'C:/Users/Moh/SemanticSearchwithApproximateNearestNeighborsandTextEmbeddings/data/Reco/ub.test', index = False, header=False)


