from __future__ import print_function
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from rdkit import Chem
from rdkit.Chem import AllChem, TorsionFingerprints
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit import DataStructs

df_all = pd.concat((pd.read_csv('train.csv'), pd.read_csv('test.csv')), axis = 0, ignore_index = True)
out_df = pd.DataFrame(df_all['smiles'])
df = df_all['smiles']
del df_all

smiles = [Chem.MolFromSmiles(x) for x in df]

import pickle

pickle.dump( smiles, open( "smiles.p", "wb" ) )
pickle.dump( df, open( "df.p", "wb" ) )