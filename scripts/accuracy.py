#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 04:59:17 2018

@author: asem
"""
import pandas as pd
from ggplot import *
import numpy as np

filename = '/media/asem/store/experimental/markovian_features/benchmarks[olfer15][psortneg][10-fold].xls'

xl_file = pd.ExcelFile(filename)

dfs = pd.read_excel(filename, sheet_name=None)

voting_performance = pd.melt( dfs['Voting'] , id_vars = ['Metric'] , value_name = 'Accuracy', var_name = 'Orders Range' )
accumulative_performance = pd.melt( dfs['Accumulative'] , id_vars = ['Metric'] , value_name = 'Accuracy', var_name = 'Orders Range' )


def adjust( df ):
    for i, row in df.iterrows():
        order_range = row['Orders Range'].strip('()')
        tokens = order_range.split(',')
        min_order = order_range.split('-')[0]
        df.loc[i,'min'] = min_order 
        df.loc[i,'max'] = 6
    return df

  

def get_theme():
    t = theme_gray()
    t._rcParams['font.size'] = 16 # Legend font size
    t._rcParams['xtick.labelsize'] = 12 # xaxis tick label size
    t._rcParams['ytick.labelsize'] = 12 # yaxis tick label size
    t._rcParams['axes.labelsize'] = 16  # axis label size
    return t


voting_performance = adjust( voting_performance )
accumulative_performance = adjust( accumulative_performance )

g = ggplot(voting_performance, aes(x='Orders Range', y='Accuracy', color='Metric')) + \
                                   geom_line() + \
                                   geom_point() +\
                                   ggtitle("Histograms Voting Classifier Performance") + get_theme() 
 
g.show()


g = ggplot(accumulative_performance, aes(x='Orders Range', y='Accuracy', color='Metric')) + \
                                   geom_line() + \
                                   geom_point() +\
                                   ggtitle("Accumulative Similarity Classifier Performance") + get_theme() 
 
g.show()
