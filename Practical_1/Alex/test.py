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

'''
df_all = pd.concat((pd.read_csv('test.csv'), pd.read_csv('test.csv')), axis = 0, ignore_index = True)
out_df = pd.DataFrame(df_all['smiles'])
df = df_all['smiles']
del df_all

Morgan = pd.concat([df, pd.DataFrame([list(Chem.AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smile), 2, nBits=2048)) for smile in df])], axis = 1)

Morgan.to_csv('Morgan2048.csv')
'''
from rdkit.Chem import MACCSkeys
import pickle

#pickle.dump( df, open( "smiles.p", "wb" ) )
df = pickle.load(open( "smiles.p", "rb" ) )

MACC = pd.concat([df, pd.DataFrame([list(MACCSkeys.GenMACCSKeys(Chem.MolFromSmiles(smile))) for smile in df])], axis = 1)
MACC.to_csv('MACC167.csv')