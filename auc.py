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

dirname = '/media/asem/store/experimental/markovian_features/cmake-build-release/src/app/[2018-08-27 19:08:20][psortNeg][order:5][grouping:olfer15]'
filename_all2minu_chi = '[eqn:differential_ALL2MIN_UNIFORM][sim:chi][auc:0.62405][n=1444][tp=0.67][fn=0.33][range=236.937][nans=0]'
filename_all2minu_cos = '[eqn:differential_ALL2MIN_UNIFORM][sim:cos][auc:0.649019][n=1444][tp=0.74][fn=0.26][range=236.937][nans=0]'
filename_all2minu_dpd1 = '[eqn:differential_ALL2MIN_UNIFORM][sim:dpd1][auc:0.594122][n=1444][tp=0.48][fn=0.52][range=236.937][nans=0]'
filename_all2minu_dpd2 = '[eqn:differential_ALL2MIN_UNIFORM][sim:dpd2][auc:0.590099][n=1444][tp=0.44][fn=0.56][range=236.937][nans=0]'
filename_all2minu_dpd3 = '[eqn:differential_ALL2MIN_UNIFORM][sim:dpd3][auc:0.582228][n=1444][tp=0.43][fn=0.57][range=236.937][nans=0]'
filename_all2minu_gaussian = '[eqn:differential_ALL2MIN_UNIFORM][sim:gaussian][auc:0.630849][n=1444][tp=0.72][fn=0.28][range=236.937][nans=0]'
filename_all2minu_intersection = '[eqn:differential_ALL2MIN_UNIFORM][sim:intersection][auc:0.610062][n=1444][tp=0.72][fn=0.28][range=236.937][nans=0]'
filename_all2minu_kl = '[eqn:differential_ALL2MIN_UNIFORM][sim:kl][auc:0.650455][n=1444][tp=0.74][fn=0.26][range=236.937][nans=0]'
files_all2minu = { 'chi': filename_all2minu_chi , 
                  'cos': filename_all2minu_cos,
                  'dpd1': filename_all2minu_dpd1,
                  'dpd2': filename_all2minu_dpd2,
                  'dpd3': filename_all2minu_dpd3,
                  'gaussian': filename_all2minu_gaussian ,
                  'intersection': filename_all2minu_intersection,
                  'kl': filename_all2minu_kl}
    
filename_all2withinu_chi = '[eqn:differential_ALL2WITHIN_UNIFORM][sim:chi][auc:0.649963][n=1444][tp=0.67][fn=0.33][range=86.5823][nans=0]'
filename_all2withinu_cos = '[eqn:differential_ALL2WITHIN_UNIFORM][sim:cos][auc:0.668784][n=1444][tp=0.74][fn=0.26][range=86.5823][nans=0]'
filename_all2withinu_dpd1 = '[eqn:differential_ALL2WITHIN_UNIFORM][sim:dpd1][auc:0.618589][n=1444][tp=0.48][fn=0.52][range=86.5823][nans=0]'
filename_all2withinu_dpd2 = '[eqn:differential_ALL2WITHIN_UNIFORM][sim:dpd2][auc:0.61679][n=1444][tp=0.44][fn=0.56][range=86.5823][nans=0]'
filename_all2withinu_dpd3 = '[eqn:differential_ALL2WITHIN_UNIFORM][sim:dpd3][auc:0.605998][n=1444][tp=0.43][fn=0.57][range=86.5823][nans=0]'
filename_all2withinu_gaussian = '[eqn:differential_ALL2WITHIN_UNIFORM][sim:gaussian][auc:0.653653][n=1444][tp=0.72][fn=0.28][range=86.5823][nans=0]'
filename_all2withinu_intersection = '[eqn:differential_ALL2WITHIN_UNIFORM][sim:intersection][auc:0.633883][n=1444][tp=0.72][fn=0.28][range=86.5823][nans=0]'
filename_all2withinu_kl = '[eqn:differential_ALL2WITHIN_UNIFORM][sim:kl][auc:0.672029][n=1444][tp=0.74][fn=0.26][range=86.5823][nans=0]'
files_all2withinu = { 'chi': filename_all2withinu_chi , 
                  'cos': filename_all2withinu_cos,
                  'dpd1': filename_all2withinu_dpd1,
                  'dpd2': filename_all2withinu_dpd2,
                  'dpd3': filename_all2withinu_dpd3,
                  'gaussian': filename_all2withinu_gaussian ,
                  'intersection': filename_all2withinu_intersection,
                  'kl': filename_all2withinu_kl}

filename_max2minu_chi = '[eqn:differential_MAX2MIN_UNIFORM][sim:chi][auc:0.63114][n=1444][tp=0.67][fn=0.33][range=270.396][nans=0]'
filename_max2minu_cos = '[eqn:differential_MAX2MIN_UNIFORM][sim:cos][auc:0.66106][n=1444][tp=0.74][fn=0.26][range=270.396][nans=0]'
filename_max2minu_dpd1 = '[eqn:differential_MAX2MIN_UNIFORM][sim:dpd1][auc:0.592501][n=1444][tp=0.48][fn=0.52][range=270.396][nans=0]'
filename_max2minu_dpd2 = '[eqn:differential_MAX2MIN_UNIFORM][sim:dpd2][auc:0.588675][n=1444][tp=0.44][fn=0.56][range=270.396][nans=0]'
filename_max2minu_dpd3 = '[eqn:differential_MAX2MIN_UNIFORM][sim:dpd3][auc:0.580101][n=1444][tp=0.43][fn=0.57][range=270.396][nans=0]'
filename_max2minu_gaussian = '[eqn:differential_MAX2MIN_UNIFORM][sim:gaussian][auc:0.641584][n=1444][tp=0.72][fn=0.28][range=270.396][nans=0]'
filename_max2minu_intersection = '[eqn:differential_MAX2MIN_UNIFORM][sim:intersection][auc:0.621761][n=1444][tp=0.72][fn=0.28][range=270.396][nans=0]'
filename_max2minu_kl = '[eqn:differential_MAX2MIN_UNIFORM][sim:kl][auc:0.663091][n=1444][tp=0.74][fn=0.26][range=270.396][nans=0]'
files_max2minu = { 'chi': filename_max2minu_chi , 
                  'cos': filename_max2minu_cos,
                  'dpd1': filename_max2minu_dpd1,
                  'dpd2': filename_max2minu_dpd2,
                  'dpd3': filename_max2minu_dpd3,
                  'gaussian': filename_max2minu_gaussian ,
                  'intersection': filename_max2minu_intersection,
                  'kl': filename_max2minu_kl}

filename_iru_chi = '[eqn:differential_informationRadius_UNIFORM][sim:chi][auc:0.640075][n=1444][tp=0.67][fn=0.33][range=405.507][nans=0]'
filename_iru_cos = '[eqn:differential_informationRadius_UNIFORM][sim:cos][auc:0.655541][n=1444][tp=0.74][fn=0.26][range=405.507][nans=0]'
filename_iru_dpd1 = '[eqn:differential_informationRadius_UNIFORM][sim:dpd1][auc:0.624578][n=1444][tp=0.48][fn=0.52][range=405.507][nans=0]'
filename_iru_dpd2 = '[eqn:differential_informationRadius_UNIFORM][sim:dpd2][auc:0.622597][n=1444][tp=0.44][fn=0.56][range=405.507][nans=0]'
filename_iru_dpd3 = '[eqn:differential_informationRadius_UNIFORM][sim:dpd3][auc:0.613236][n=1444][tp=0.43][fn=0.57][range=405.507][nans=0]'
filename_iru_gaussian = '[eqn:differential_informationRadius_UNIFORM][sim:gaussian][auc:0.639796][n=1444][tp=0.72][fn=0.28][range=405.507][nans=0]'
filename_iru_intersection = '[eqn:differential_informationRadius_UNIFORM][sim:intersection][auc:0.612661][n=1444][tp=0.72][fn=0.28][range=405.507][nans=0]'
filename_iru_kl = '[eqn:differential_informationRadius_UNIFORM][sim:kl][auc:0.65575][n=1444][tp=0.74][fn=0.26][range=405.507][nans=0]'
files_iru = { 'chi': filename_iru_chi , 
                  'cos': filename_iru_cos,
                  'dpd1': filename_iru_dpd1,
                  'dpd2': filename_iru_dpd2,
                  'dpd3': filename_iru_dpd3,
                  'gaussian': filename_iru_gaussian ,
                  'intersection': filename_iru_intersection,
                  'kl': filename_iru_kl}

def make_dataframe_from_files( dictionary , dirname ):
    x_axis = np.arange(0,1,0.01)
    new_dict = { 'fpr' : x_axis }

    for k,v in dictionary.items():
        input = np.loadtxt( dirname + '/' + v , dtype = np.float32 )
        predictions = input[0,:]
        scores = input[1,:]
        fpr, tpr, thresholds = metrics.roc_curve(predictions, scores )
        auc = metrics.auc( fpr , tpr )
        x_args = np.abs(np.subtract.outer(x_axis, fpr)).argmin(1)
        y_axis = tpr[x_args]
        label = '{0} (AUC={1:0.2f})'.format(k,auc)
        new_dict[ label ] = y_axis
        
        
    df = pd.DataFrame(new_dict)
    return pd.melt( df , id_vars = ['fpr'] , value_name = 'tpr' , var_name ='metric' )


df = make_dataframe_from_files(files_all2minu , dirname )
g = ggplot(df, aes(x='fpr', y='tpr', color='metric')) + geom_line() + \
 ggtitle("All-to-min (UNIFORM)")
g.show()

df = make_dataframe_from_files(files_all2withinu , dirname )
g = ggplot(df, aes(x='fpr', y='tpr', color='metric')) + geom_line() + \
 ggtitle("All-to-within (UNIFORM)")
g.show()

df = make_dataframe_from_files(files_max2minu , dirname )
g = ggplot(df, aes(x='fpr', y='tpr', color='metric')) + geom_line() + \
 ggtitle("Max-to-min (UNIFORM)")
g.show()

df = make_dataframe_from_files(files_iru , dirname )
g = ggplot(df, aes(x='fpr', y='tpr', color='metric')) + geom_line() + \
 ggtitle("Information Radius (Jensen-Shannon) (UNIFORM)")
g.show()