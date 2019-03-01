import pandas as pd
from ggplot import *
import numpy as np
import itertools

metrics = {'cos' : 'Cosine',
               'kl' : 'Kullback-Leibler',
               'chi' : 'Chi-squared',
               'dpd1' : 'DPD (α=1)',
               'dpd2' : 'DPD (α=2)', 
               'dpd3' : 'DPD (α=3)',
               'gaussian' : 'Gaussian RBF' , 
               'bhat' : 'Bhattacharyya', 
               'hell' : 'Hellinger' }
    
orders_ranges = ['2',
            '(2-3)',
            '(2-4)',
            '(2-5)',
            '(2-6)',
            '3',
            '(3-4)',
            '(3-5)',
            '(3-6)',
            '4',
            '(4-5)',
            '(4-6)',
            '5',
            '(5-6)',
            '6']
    
def extract_results( fname ):

    
    rows = metrics.values()
    
    means = pd.DataFrame(index = rows  )
    stds = pd.DataFrame(index = rows )
    
    means.index.name = 'Metric'
    stds.index.name = 'Metric'
    
    with open(fname) as f:
        items = f.read().split('[Params]')
        items = items[1:]
        for item in items:
            first_line = item.split('\n')[0].strip()
            key_vals = [ kv.strip('][') for kv in first_line.split('[', 1)[1].split(']') 
            if 'order' in first_line ]
#            print(key_vals)
#            key_vals = key_vals[1:]
            order = key_vals[0].split(':')[1]
            if order.split('-')[0] == order.split('-')[1]:
                order = order.split('-')[0]
            else :
                order = "({})".format(order)
            
            metric = key_vals[1].split(':')[1]
            
            for kv in item.splitlines():
                if( 'Overall Accuracy' in kv ):
                    result = kv.split(':')[1].split()
                    mean = result[0]
                    std = result[1].strip('()±')
#                    print("({},{}):{})".format(order,metrics[metric],mean))
#                    print(std)
#                    print(kv)
                    means.loc[metrics[metric],'Metric'] = metrics[metric]
                    means.loc[metrics[metric],order] = mean
                    stds.loc[metrics[metric],order] = std

#    
#    for i, row in means.iterrows():
#        order_range = row['Orders Range'].strip('()')
#        tokens = order_range.split(',')
#        min_order = order_range.split('-')[0]
#        df.loc[i,'min'] = min_order 
#        df.loc[i,'max'] = 6
#

    return means,stds

def get_theme():
    t = theme_gray()
    t._rcParams['font.size'] = 16 # Legend font size
    t._rcParams['xtick.labelsize'] = 12 # xaxis tick label size
    t._rcParams['ytick.labelsize'] = 12 # yaxis tick label size
    t._rcParams['axes.labelsize'] = 16  # axis label size
    return t


fname_ofer15_voting = '/media/asem/store/experimental/markovian_features/ofer15-voting.txt'
fname_ofer15_acc = '/media/asem/store/experimental/markovian_features/ofer15-acc.txt'
fname_nogrouping_voting = '/media/asem/store/experimental/markovian_features/nogrouping-voting.txt'
fname_nogrouping_acc = '/media/asem/store/experimental/markovian_features/nogrouping-acc.txt'

ofer15_voting_means, ofer15_voting_stds = extract_results(fname_ofer15_voting)
ofer15_acc_means, ofer15_acc_stds = extract_results(fname_ofer15_voting)
nogrouping_voting_means, nogrouping_voting_stds = extract_results(fname_ofer15_voting)
nogrouping_acc_means, nogrouping_acc_stds = extract_results(fname_ofer15_voting)

test = pd.melt( ofer15_voting_means , id_vars = ['Metric'] , value_name = 'Accuracy', var_name = 'Orders Range'  )

# Standard deviation
g = ggplot(test , aes(x='Orders Range',y='Accuracy',color='Metric',fill='Metric')) +\
    geom_bar( stat="identity") + facet_wrap('Orders Range') +\
  ggtitle("using standard deviation")
#  geom_errorbar( aes(x=Species, ymin=mean-sd, ymax=mean+sd), width=0.4, colour="orange", alpha=0.9, size=1.5) +

 
    

g.show()
