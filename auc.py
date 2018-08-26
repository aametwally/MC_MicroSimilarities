#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 01:36:24 2018

@author: asem
"""

import numpy as np
from sklearn import metrics
from ggplot import *
import pandas as pd

dirname = '/media/asem/store/experimental/markovian_features/cmake-build-release/src/app/[psortNeg][order:4][grouping:olfer15]'
filename = '[eqn:ALL2MIN_UNIFORM][sim:chi][auc:0.49282][n=1444][tp=0.29][fn=0.71][range=255.637][nans=0]'

input = np.loadtxt( dirname + '/' + filename , dtype = np.float32 )
predictions = input[0,:]
scores = input[1,:]


fpr, tpr, thresholds = metrics.roc_curve(predictions, scores )

df = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
ggplot(df, aes(x='fpr', y='tpr')) + geom_line() + geom_abline(linetype='dashed')
