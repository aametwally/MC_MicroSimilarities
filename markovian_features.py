import numpy as np
import os
from sklearn.decomposition import PCA

path = '/media/asem/store/experimental/build-markovian_features-Desktop_Qt_5_9_1_GCC_64bit-Release/src/app/data'

consensus_gprofile_files = sorted([ f for f in os.listdir(path) \
                              if f.startswith('consensus_gprofile') ])
consensus_profile_files = sorted([ f for f in os.listdir(path) \
                              if f.startswith('consensus_profile') ])

test_files = [ f for f in os.listdir(path)  if f.startswith('test')]

test_gprofile_files = sorted( list( filter( lambda s : 'gprofile' in s, test_files )))
test_profile_files = sorted(list( filter( lambda s : '_profile' in s, test_files )))

f2array = lambda f: np.loadtxt( path + '/' + f, dtype = np.float32 )
array2vec = lambda a : a.reshape(1,-1)
f2label = lambda f: int(f.split('.')[0].split('_')[-1])

consensus_profiles_gen = map( array2vec , map( f2array , consensus_profile_files ))
consensus_gprofiles_gen = map( array2vec, map( f2array , consensus_gprofile_files ))
test_profiles_gen = map( f2array , test_profile_files )
test_gprofiles_gen = map( f2array , test_gprofile_files )
test_labels = np.fromiter( map( f2label , test_profile_files ) , dtype = np.int32 ).reshape(1,-1)

def canonical_features_array( features ):
    population = None
    try:
        population = next( features ).reshape( 1 , -1 )
    except StopIteration:
        pass
    for f in features:
        population = np.vstack( (population , f.reshape( 1 , -1 )))
    
    return population

# Expensive Operations
consensus_profiles = canonical_features_array( consensus_profiles_gen )
consensus_gprofiles = canonical_features_array( consensus_gprofiles_gen )
test_profiles = canonical_features_array( test_profiles_gen )
test_gprofiles = canonical_features_array( test_gprofiles_gen )


pca_p = PCA( n_components = 0.95 )
pca_p.fit( consensus_profiles )
ev_p = pca_p.explained_variance_ratio_.cumsum()


pca_gp = PCA( n_components = 0.95 )
pca_gp.fit( consensus_gprofiles )
ev_gp = pca_gp.explained_variance_ratio_.cumsum()


train_p_reduced = pca_p.transform( consensus_profiles )
train_gp_reduced = pca_gp.transform( consensus_gprofiles )
train_labels = np.range( 1 , train_p_reduced.shape[0] + 1 , np.int32 )


test_p_reduced = pca_p.transform( test_profiles )
test_gp_reduced = pca_gp.transform( test_gprofiles )
