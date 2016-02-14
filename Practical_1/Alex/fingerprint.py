from __future__ import print_function
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#from sklearn.linear_model import LinearRegression
#from sklearn.ensemble import RandomForestRegressor
from rdkit import Chem
#from rdkit.Chem import AllChem, TorsionFingerprints
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import DataStructs
from rdkit.Chem import AllChem

df_all = pd.concat((pd.read_csv("train.csv"), pd.read_csv("test.csv")), axis=0, ignore_index = True)
out_df = pd.DataFrame(df_all['smiles'])
df = df_all['smiles']
del df_all

def subset_iter(df, iteration, root):
    Morgan = pd.concat([df
    	, pd.DataFrame(
    		[list(Chem.AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile), 2, nbits=2048)) for smile in df])]
    	, axis = 1)
    fname = root + str(iteration) + ".csv"
    Morgan.to_csv(fname)
    
def iter_loop(df, root, start = 0):
    i = 1
    df = df[start:]
    while(len(df) > 0):
        subset_iter(df[0:200000], i, root)
        df = df[200000:]
        print('run' + i + ' complete')
        i+=1
    
iter_loop(df, 'Morgan', start = 0)