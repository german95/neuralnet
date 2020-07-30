# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 15:07:16 2020

@author: massa
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import autoreload
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

#starts from the .py location

train_ds = pd.read_csv('data/train.csv')
to_drop = pd.read_csv('mtcars.txt')
to_drop = to_drop.values.tolist()

to_drop_unlist = []
for lista in to_drop:
    to_drop_unlist.append(lista[0])

train_ds = train_ds.drop(to_drop_unlist, axis=1)
train_ds.head()

train_ds['nac'] = train_ds['nac'].astype('category')
#train_ds['TARGET'] = train_ds['TARGET'].astype('category')


dummies = pd.get_dummies(train_ds['nac'], prefix='nac', drop_first=False)
train_ds = train_ds.drop('nac', axis=1)
train_ds = pd.concat([train_ds, dummies], axis=1)

train_ds['age'] = train_ds['age'].mask(train_ds['age'] < 20, 20)

#ESCALO : 

columnas = train_ds.columns
quant_features = columnas[~columnas.str.contains("nac") * ~columnas.str.contains("TARGET")]
# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in quant_features:
    mean, std = train_ds[each].mean(), train_ds[each].std()
    scaled_features[each] = [mean, std]
    train_ds.loc[:, each] = (train_ds[each] - mean)/std


# Save data for approximately the last 21 days 
test_data = train_ds[-1000:]
train_ds = train_ds[:-1000]


train_ds['ANTITARGET'] = 1 - train_ds['TARGET']
test_data['ANTITARGET'] = 1 - test_data['TARGET']


# Class count
count_class_0, count_class_1 = train_ds.TARGET.value_counts()

df_class_0 = train_ds[train_ds['TARGET'] == 0]
df_class_1 = train_ds[train_ds['TARGET'] == 1]

df_class_0_under = df_class_0.sample(count_class_1)
df_under = pd.concat([df_class_0_under, df_class_1], axis=0)

#train_ds = df_under
train_ds = train_ds2
train_ds = train_ds.reset_index(drop=True)

# Separate the data into features and targets
target_fields = ['TARGET','ANTITARGET']
features, targets = train_ds.drop(target_fields, axis=1), train_ds[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

#Hold out the last 60 days or so of the remaining data as a validation set
train_features, train_targets = features[:-3000], targets[:-3000]
val_features, val_targets = features[-3000:], targets[-3000:]


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 1000
learning_rate = 0.03
hidden_nodes = 1000
output_nodes = 2


def MSE(y, Y):
    return np.mean((y-Y)**2)



network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)
N_i = train_features.shape[1]


losses = {'train':[], 'validation':[]}
for ii in range(iterations):
    # Go through a random batch of 128 records from the training data set
    batch = np.random.choice(train_features.index, size=128)
    X, y = train_features.iloc[batch].values, train_targets.iloc[batch].values
            
    
    network.train(X, y)
        
    # Printing out the training progress
    train_loss = MSE(network.run(train_features), train_targets.values)
    val_loss = MSE(network.run(val_features), val_targets.values)
    sys.stdout.write("\rProgress: {:2.1f}".format(100 * ii/float(iterations)) \
                     + "% ... Training loss: " + str(train_loss)[:5] \
                     + " ... Validation loss: " + str(val_loss)[:5])
    sys.stdout.flush()
    
    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)


#from sklearn import metrics

fpr, tpr, thresholds = metrics.roc_curve(train_targets['TARGET'].values, network.run(train_features).T[0])
metrics.auc(fpr, tpr)


fpr, tpr, thresholds = metrics.roc_curve(val_targets['TARGET'].values, network.run(val_features).T[0])
metrics.auc(fpr, tpr)

fpr, tpr, thresholds = metrics.roc_curve(test_targets['TARGET'].values, network.run(test_features).T[0])
metrics.auc(fpr, tpr)




