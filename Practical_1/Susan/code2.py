# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import Ridge
# from sklearn import cross_validation
# from sklearn.decomposition import PCA

# #from rdkit import Chem

# def write_to_file(filename, predictions):
#     with open(filename, "w") as f:
#         f.write("Id,Prediction\n")
#         for i,p in enumerate(predictions):
#             f.write(str(i+1) + "," + str(p) + "\n")



# def RMSE(predictions,labels):
#     sum = 0
#     for i in range(len(labels)):
#         sum += (predictions[i]-labels[i])**2
#     return (sum/len(labels))**0.5

# df_train = pd.read_csv("train.csv")
# df_train1 = pd.read_csv("newfeature\SimilarityV1.csv")
# df_train2 = pd.read_csv("newfeature\Bag1.csv")
# df_train3 = pd.read_csv("newfeature\Bag2.csv")
# df_train4 = pd.read_csv("newfeature\Bag3.csv")
# df_train5 = pd.read_csv("newfeature\Bag4.csv")
# df_train6 = pd.read_csv("newfeature\Bag5.csv")
# df_test = pd.read_csv("test.csv")

# #store gap values
# Y_train = df_train.gap.values
# #row where testing examples start
# test_idx = df_train.shape[0]
# #delete 'Id' column
# df_test = df_test.drop(['Id'], axis=1)
# #delete 'gap' column
# df_train = df_train.drop(['gap'], axis=1)


# df_all = pd.concat((df_train, df_test), axis=0)
# df_all = df_all.reset_index()
# df_all = df_all.drop(['smiles'], axis = 1)
# df_all1 = df_train1.drop(['smiles'], axis=1)
# df_all = pd.concat((df_all, df_all1), axis=1)
# df_all2  = df_train2.drop(['smiles'], axis=1)
# df_all = pd.concat((df_all, df_all2), axis = 1)
# df_all3  = df_train3.drop(['smiles'], axis=1)
# df_all = pd.concat((df_all, df_all3), axis = 1)
# df_all4 = df_train4.drop(['smiles'], axis = 1)
# df_all = pd.concat((df_all, df_all4), axis = 1)
# df_all5 = df_train5.drop(['smiles'], axis = 1)
# df_all = pd.concat((df_all, df_all5), axis = 1)

# X_train = df_all.iloc[:test_idx]
# X_test = df_all.iloc[test_idx:]

# X_train = df_all.iloc[:test_idx]
# X_test = df_all.iloc[test_idx:]
# Y_train = pd.read_csv("Y_train.csv")

# # X_temp = X_train.iloc[:1000]
# # X_temp.to_csv("X_temp.csv")
# # Y_temp = Y_train.iloc[:1000]
# # Y_temp.to_csv("Y_temp.csv")



# Y_train = Y_train.drop("Id", axis = 1)
# Y_train = Y_train.values
# X_mytrain, X_mytest, Y_mytrain, Y_mytest = cross_validation.train_test_split(X_train, Y_train, test_size=0.7, random_state=1)

# RF = RandomForestRegressor()
# RF.fit(X_mytrain, Y_mytrain)
# RF_pred = RF.predict(X_test)
# write_to_file("RF_output1.csv", RF_pred)
# # RF = RandomForestRegressor()
# # RF.fit(X_mytrain, Y_mytrain)
# # RF_pred = RF.predict(X_mytest)
# # RMSE(RF_pred, Y_mytest)
# # write_to_file("RF_output.csv", RF_pred)
# # pca = PCA().fit(X_mytrain)
# # X_mytrain_pca = pca.transform(X_mytrain)
# # pca = PCA().fit(X_mytest)
# # X_mytest_pca = pca.transform(X_mytest)

# # RF = RF.fit(X_mytrain_pca, Y_mytrain)
# # PCA_pred = RF.predict(X_mytest_pca)
# # print RMSE(PCA_pred, Y_mytest)

# # pca = PCA().fit(X_test)
# # X_test_pca = pca.transform(X_test)
# # PCA_pred = RF.predict(X_test)
# # write_to_file("pca_output.csv", PCA_pred)




import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn import cross_validation
from sklearn.decomposition import PCA

#from rdkit import Chem

def write_to_file(filename, predictions):
    with open(filename, "w") as f:
        f.write("Id,Prediction\n")
        for i,p in enumerate(predictions):
            f.write(str(i+1) + "," + str(p) + "\n")



def RMSE(predictions,labels):
    sum = 0
    for i in range(len(labels)):
        sum += (predictions[i]-labels[i])**2
    return (sum/len(labels))**0.5

df_all = pd.read_csv("df_all.csv")
test_idx = 1000000

X_train = df_all.iloc[:test_idx]
Y_train = pd.read_csv("Y_train.csv")
Y_train = Y_train.drop("Id", axis = 1)
Y_train = Y_train.values

# X_test = df_all.iloc[test_idx:]
# X_test.to_csv("X_test.csv")
X_test = pd.read_csv("X_test.csv")


RF = RandomForestRegressor()
X_mytrain, X_mytest, Y_mytrain, Y_mytest = cross_validation.train_test_split(X_train, Y_train, test_size=0.7, random_state=1)

# X_temp =pd.read_csv("X_temp.csv")
# Y_temp = pd.read_csv("Y_temp.csv")
# Y_temp=Y_temp.loc[:, "Prediction"]
# Y_temp=Y_temp.values

RF = RF.fit(X_mytrain, Y_mytrain)
RF_pred = RF.predict(X_test)

write_to_file("RF_output2.csv", RF_pred)

#print type(PCA_pred)
#print type(Y_mytest)
#print PCA_pred.shape
#, Y_mytest.shape
# print PCA_pred
# print RMSE(PCA_pred, Y_mytest)
# print pca.explained_variance_ratio_