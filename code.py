#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 13:21:39 2021

@author: chenwynn
"""

#%%
import pandas as pd
import numpy as np

data = pd.read_csv('data.csv')
data['date'] = pd.to_datetime(data['date'])

stk_freq = data['ticker'].value_counts()

#%%

'''
Most of the stocks have 2005 trading dates' data and 248 different stocks, but some of them don't. We have to determine 
if these stocks have missing data or they disappear or appear later. Try to eliminate those stocks without efficient data.
'''

#%%
data_e = data.copy(deep = True)
data_e.set_index(['ticker'], inplace = True)
stk_freq.index.name = 'ticker'
data_e = data_e.loc[stk_freq[stk_freq == 2005].index]
data_e.reset_index(inplace = True)

import matplotlib.pyplot as plt

plt.figure()
plt.plot(data['date'].value_counts().sort_index())

plt.figure()
plt.plot(data_e['date'].value_counts().sort_index())
#%%

'''
The stocks number is now consistent with time.
Try to set up several indexes that are informative about stock returns.
'''

#%%
# informative plot
from matplotlib import colors
import matplotlib as mpl

cmap = mpl.cm.get_cmap('RdBu')

def quantile_plot(index, rets):
    ## input: pd.series, pd.dataframe, both with index
    ## output: plots
    
    quantile = pd.qcut(index, 10, list(range(1, 11)))
    quantile.name = 'q'
    rets = rets.join(quantile)
    ret_plot = rets.groupby(by = 'q').mean()
    
    plt.figure()
    for i in range(1, 11):
        plt.plot(ret_plot.loc[i], c = cmap(i/10))
    
    return 0

#%%

#%%
# return dataframe
data_e.set_index(['ticker', 'date'], inplace = True)
for i in range(1, 11):
    data_e['ret_'+str(i)] = (np.exp(np.log(data_e['last']).groupby('ticker').diff(i))-1).groupby('ticker').shift(-i)
    
#%%
'''
See if the volume alone is informative
'''
#%%
quantile_plot(data_e['volume'], data_e.iloc[:, 2:11])
quantile_plot((data_e['volume']/data_e['volume'].groupby('date').sum()), data_e.iloc[:, 2:11]-(data_e.iloc[:, 2:11].groupby('date').mean()))
quantile_plot(data_e['volume'].groupby('ticker').diff(), data_e.iloc[:, 2:11]-(data_e.iloc[:, 2:11].groupby('date').mean()))

#%%

'''
The volume index alone is more informative on the time intercept of the stock 
return, it might not be a good cross-sectional signal to select stocks. 
Specifically, the volume difference is informative about asset returns on first
two quantile stocks. And the regularized volume is not informative about asset 
returns.

Next, see if the return alone is informative
'''

#%%
for i in range(1, 11):
    data_e['ret_'+str(-i)] = (np.exp(np.log(data_e['last']).groupby('ticker').diff(i))-1)

quantile_plot(data_e['ret_-10'], data_e.iloc[:, 2:11])
quantile_plot(data_e['ret_-10']-(data_e['ret_10'].groupby('date').mean()), data_e.iloc[:, 2:11]-(data_e.iloc[:, 2:11].groupby('date').mean()))

#%%
'''
Return index performs differently on the time intercept and the cross-sectional part. 
For cross-sectional part, a reverse trend is clear for the last quantile stocks.
For the time part, there's a momentum trend.
Use both the volume difference and the regularized past ten day return to fit a
lightGBM model and generate signals for trading, y is the forward one day return 
of the stock.
'''

#%%
import lightgbm as lgb

data_e['sig1'] = data_e['volume'].groupby('ticker').diff()
data_e['sig2'] = data_e['ret_-10']-(data_e['ret_10'].groupby('date').mean())

train_x = data_e.reset_index().set_index(['date','ticker']).sort_index()[:pd.to_datetime('2020-01-04')].loc[:,'sig1':'sig2']
train_y = data_e.reset_index().set_index(['date','ticker']).sort_index()[:pd.to_datetime('2020-01-04')]['ret_1']
test_x = data_e.reset_index().set_index(['date','ticker']).sort_index()[pd.to_datetime('2020-01-04'):].loc[:,'sig1':'sig2']
test_y = data_e.reset_index().set_index(['date','ticker']).sort_index()[pd.to_datetime('2020-01-04'):]['ret_1']

train_data = lgb.Dataset(data=train_x,label=train_y)
test_data = lgb.Dataset(data=test_x,label=test_y)

num_round = 10
param = {'num_leaves':21, 'num_trees':100, 'objective':'regression'}
bst = lgb.train(param, train_data, num_round, valid_sets=[test_data])

ytrain = pd.Series(bst.predict(train_x, num_iteration=bst.best_iteration), index=train_y.index)
ypred = pd.Series(bst.predict(test_x, num_iteration=bst.best_iteration), index = test_y.index)

quantile_plot(ytrain, data_e.reset_index().set_index(['date','ticker']).sort_index()[:pd.to_datetime('2020-01-04')].iloc[:, 2:11])
quantile_plot(ypred, data_e.reset_index().set_index(['date','ticker']).sort_index()[pd.to_datetime('2020-01-04'):].iloc[:, 2:11])

#%%
'''
from plot, the result is pretty good surprisingly. Now construct a portfolio 
consists by forty stocks with highest ypred, the consturction weight is equally
weighted.
'''

#%%
ypred.name = 'ypred'
port = ypred.to_frame()
port['rank'] = port.groupby('date').rank('min')
port['hold'] = port['rank'] > 160

port['return'] = data_e.reset_index().set_index(['date','ticker']).sort_index()[pd.to_datetime('2020-01-04'):].loc[:,'ret_1']
port.dropna(inplace = True)

holdings = port[port['hold']]

import math

#PnL
pnl = (holdings['return'].groupby('date').mean()+1).cumprod() - 1
# Annualized Volume
Vol = np.std(holdings['return'].groupby('date').mean()) * math.sqrt(252)
# sharpe ratio
sharpe = (np.mean(holdings['return'].groupby('date').mean()) - 0.0004767)/np.std(holdings['return'].groupby('date').mean())*math.sqrt(252)

#%%
'''
I intended to add maximum drawdown, sortino ratio and annualized turnover to evaluate
the strategy. And I meant to add a time chosen function using LSTM. But the time is too short lol
'''

#%%
# Define LSTM Neural Networks
import torch
from torch import nn

class LstmRNN(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """
    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()
 
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers) # utilize the LSTM model in torch.nn 
        self.forwardCalculation = nn.Linear(hidden_size, output_size)
 
    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s*b, h)
        x = self.forwardCalculation(x)
        x = x.view(s, b, -1)
        return x
    












