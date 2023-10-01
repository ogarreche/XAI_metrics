
#---------------------------------------------------------------------
# Importing Libraries
print('---------------------------------------------------------------------------------')
print('Importing Libraries')
print('---------------------------------------------------------------------------------')
print('')

import numpy
import time
import tensorflow as tf
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn.ensemble import RandomForestClassifier
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.preprocessing import sequence
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from imblearn.over_sampling import RandomOverSampler
import shap
from scipy.special import softmax
np.random.seed(0)
from sklearn.model_selection import train_test_split
import sklearn
#---------------------------------------------------------------------
# Defining metric equations

print('---------------------------------------------------------------------------------')
print('Defining Metric Equations')
print('---------------------------------------------------------------------------------')
print('')
def print_feature_importances_shap_values(shap_values, features):
    '''
    Prints the feature importances based on SHAP values in an ordered way
    shap_values -> The SHAP values calculated from a shap.Explainer object
    features -> The name of the features, on the order presented to the explainer
    '''
    # Calculates the feature importance (mean absolute shap value) for each feature
    importances = []
    for i in range(shap_values.values.shape[1]):
        importances.append(np.mean(np.abs(shap_values.values[:, i])))
    # Calculates the normalized version
    importances_norm = softmax(importances)
    # Organize the importances and columns in a dictionary
    feature_importances = {fea: imp for imp, fea in zip(importances, features)}
    feature_importances_norm = {fea: imp for imp, fea in zip(importances_norm, features)}
    # Sorts the dictionary
    feature_importances = {k: v for k, v in sorted(feature_importances.items(), key=lambda item: item[1], reverse = True)}
    feature_importances_norm= {k: v for k, v in sorted(feature_importances_norm.items(), key=lambda item: item[1], reverse = True)}
    # Prints the feature importances
    for k, v in feature_importances.items():
        print(f"{k} -> {v:.4f} (softmax = {feature_importances_norm[k]:.4f})")


def ACC(TP,TN,FP,FN):
    Acc = (TP+TN)/(TP+FP+FN+TN)
    return Acc
def ACC_2 (TP, FN):
    ac = (TP/(TP+FN))
    return ac
def PRECISION(TP,FP):
    Precision = TP/(TP+FP)
    return Precision
def RECALL(TP,FN):
    Recall = TP/(TP+FN)
    return Recall
def F1(Recall, Precision):
    F1 = 2 * Recall * Precision / (Recall + Precision)
    return F1
def BACC(TP,TN,FP,FN):
    BACC =(TP/(TP+FN)+ TN/(TN+FP))*0.5
    return BACC
def MCC(TP,TN,FP,FN):
    MCC = (TN*TP-FN*FP)/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**.5)
    return MCC
def AUC_ROC(y_test_bin,y_score):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    auc_avg = 0
    counting = 0
    for i in range(n_classes):
      fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
      auc_avg += auc(fpr[i], tpr[i])
      counting = i+1
    return auc_avg/counting

def oversample(X_train, y_train):
    oversample = RandomOverSampler(sampling_strategy='minority')
    # Convert to numpy and oversample
    x_np = X_train.to_numpy()
    y_np = y_train.to_numpy()
    x_np, y_np = oversample.fit_resample(x_np, y_np)

    # Convert back to pandas
    x_over = pd.DataFrame(x_np, columns=X_train.columns)
    y_over = pd.Series(y_np)
    return x_over, y_over

#---------------------------------------------------------------------


print('---------------------------------------------------------------------------------')
print('Generating Sparsity Graph')
print('---------------------------------------------------------------------------------')
print('')

x_axis_cic = [0, 5, 10, 20, 40, 70] 
x_axis_sml = [0, 5, 10, 20] 
# CICIDS
#SHAP
#RF 
y_axis_RF = [0.979, 0.969, 0.977, 0.971, 0.967, 0.875]
#DNN 
y_axis_DNN = [0.755, 0.755, 0.755, 0.823, 0.755, 0.755]
#LGBM
y_axis_LGBM =  [0.999, 0.995, 0.995, 0.995, 0.991, 0.988]

y_axis_SVM = [0.962, 0.936, 0.918, 0.896, 0.752, 0.754]

#ADA
y_axis_ADA = [0.752, 0.762, 0.762, 0.751, 0.752, 0.752]
#KNN
y_axis_KNN =  [0.999, 0.997, 0.999,0.999,0.999,0.999]
#MLP
y_axis_MLP = [0.714, 0.715, 0.715, 0.716, 0.718, 0.755]

plt.clf()

# Plot the first line
plt.plot(x_axis_cic, y_axis_RF, label='RF', color='blue', linestyle='--', marker='o')

# Plot the second line
plt.plot(x_axis_cic, y_axis_DNN, label='DNN', color='red', linestyle='--', marker='x')

# Plot the third line
plt.plot(x_axis_cic, y_axis_LGBM, label='LGBM', color='green', linestyle='--', marker='s')

# Plot the fourth line
plt.plot(x_axis_cic, y_axis_SVM, label='SVM', color='purple', linestyle='--', marker='p')

# Plot the fifth line
plt.plot(x_axis_cic, y_axis_MLP, label='MLP', color='orange', linestyle='--', marker='h')

# Plot the sixth line
# plt.plot(x_axis_cic, y_axis_KNN, label='KNN', color='magenta', linestyle='--', marker='+')

# Plot the seventh line
plt.plot(x_axis_cic, y_axis_ADA, label='ADA', color='cyan', linestyle='--', marker='_')

# Enable grid lines (both major and minor grids)
plt.grid()

# Customize grid lines (optional)
# plt.grid()

# Add labels and a legend
plt.xlabel('Features')
plt.ylabel('Accuracy')
plt.legend()

# Set the title of the plot
# plt.title('Accuracy x Features - SHAP CIC')

# Show the plot
plt.show()
plt.savefig('GRAPH_ACC_SHAP_CIC.png')
plt.clf()

print('AUC - CIC SHAP')

auc = np.trapz(y_axis_RF, x_axis_cic)/x_axis_cic[-1]
print('RF')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_DNN, x_axis_cic)/x_axis_cic[-1]
print('DNN')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_LGBM,  x_axis_cic)/x_axis_cic[-1]
print('LGBM')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_SVM, x_axis_cic)/x_axis_cic[-1]
print('SVM')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_KNN,  x_axis_cic)/x_axis_cic[-1]
print('KNN')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_ADA,  x_axis_cic)/x_axis_cic[-1]
print('ADA')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_MLP,  x_axis_cic)/x_axis_cic[-1]
print('MLP')
print(f"AUC: {auc}")


###########################################################################################################################################################################################################################################################################################################################
#SIMARGL
#SHAP
    #RF
y_axis_RF =  [0.997, 0.995, 0.941, 0.997]
    #DNN
y_axis_DNN =  [0.776, 0.555, 0.53, 0.555]
    #LGBM
y_axis_LGBM = [0.999, 0.999, 0.998, 0.999]
    #SVM
y_axis_SVM =  [0.932, 0.902, 0.924, 0.941]
    #ADA
y_axis_ADA = [0.763, 0.758, 0.636, 0.67]
    #KNN
y_axis_KNN = [0.999,0.998,0.998,0.998]
    #MLP
y_axis_MLP =  [0.714, 0.556, 0.555, 0.555]

plt.clf()

# Plot the first line
plt.plot(x_axis_sml, y_axis_RF, label='RF', color='blue', linestyle='--', marker='o')

# Plot the second line
plt.plot(x_axis_sml, y_axis_DNN, label='DNN', color='red', linestyle='--', marker='x')

# Plot the third line
plt.plot(x_axis_sml, y_axis_LGBM, label='LGBM', color='green', linestyle='--', marker='s')

# Plot the fourth line
plt.plot(x_axis_sml, y_axis_SVM, label='SVM', color='purple', linestyle='--', marker='p')

# Plot the fifth line
plt.plot(x_axis_sml, y_axis_MLP, label='MLP', color='orange', linestyle='--', marker='h')

# Plot the sixth line
plt.plot(x_axis_sml, y_axis_KNN, label='KNN', color='magenta', linestyle='--', marker='+')

# Plot the seventh line
plt.plot(x_axis_sml, y_axis_ADA, label='ADA', color='cyan', linestyle='--', marker='_')

# Enable grid lines (both major and minor grids)
plt.grid()

# Customize grid lines (optional)
# plt.grid()

# Add labels and a legend
plt.xlabel('Features')
plt.ylabel('Accuracy')
plt.legend()

# Set the title of the plot
# plt.title('Accuracy x Features - SHAP SML')

# Show the plot
plt.show()
plt.savefig('GRAPH_ACC_SHAP_SML.png')
plt.clf()

print('AUC - SML SHAP')

auc = np.trapz(y_axis_RF, x_axis_sml)/x_axis_sml[-1]
print('RF')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_DNN, x_axis_sml)/x_axis_sml[-1]
print('DNN')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_LGBM, x_axis_sml)/x_axis_sml[-1]
print('LGBM')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_SVM, x_axis_sml)/x_axis_sml[-1]
print('SVM')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_KNN, x_axis_sml)/x_axis_sml[-1]
print('KNN')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_ADA, x_axis_sml)/x_axis_sml[-1]
print('ADA')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_MLP, x_axis_sml)/x_axis_sml[-1]
print('MLP')
print(f"AUC: {auc}")


###########################################################################################################################################################################################################################################################################################################################
#CICIDS
#LIME

y_axis_RF = [0.979, 0.984, 0.985, 0.957, 0.98, 0.983]
y_axis_DNN = [0.755, 0.755, 0.741, 0.777, 0.783, 0.819]
y_axis_LGBM= [0.999, 0.999, 0.999, 0.994, 0.999, 0.999]
y_axis_ADA  = [0.751, 0.83, 0.817, 0.824, 0.857, 0.75]
y_axis_MLP  = [0.714, 0.714, 0.755, 0.716, 0.715, 0.714]
y_axis_SVM = [0.961, 0.827, 0.894, 0.907, 0.952, 0.961]
y_axis_KNN = [0.999, 0.985, 0.997, 0.998, 0.999, 0.999]

plt.clf()

# Plot the first line
plt.plot(x_axis_cic, y_axis_RF, label='RF', color='blue', linestyle='--', marker='o')

# Plot the second line
plt.plot(x_axis_cic, y_axis_DNN, label='DNN', color='red', linestyle='--', marker='x')

# Plot the third line
plt.plot(x_axis_cic, y_axis_LGBM, label='LGBM', color='green', linestyle='--', marker='s')

# Plot the fourth line
plt.plot(x_axis_cic, y_axis_SVM, label='SVM', color='purple', linestyle='--', marker='p')

# Plot the fifth line
plt.plot(x_axis_cic, y_axis_MLP, label='MLP', color='orange', linestyle='--', marker='h')

# Plot the sixth line
plt.plot(x_axis_cic, y_axis_KNN, label='KNN', color='magenta', linestyle='--', marker='+')

# Plot the seventh line
plt.plot(x_axis_cic, y_axis_ADA, label='ADA', color='cyan', linestyle='--', marker='_')

# Enable grid lines (both major and minor grids)
plt.grid()

# Customize grid lines (optional)
# plt.grid()

# Add labels and a legend
plt.xlabel('Features')
plt.ylabel('Accuracy')
plt.legend()

# Set the title of the plot
# plt.title('Accuracy x Features - LIME CIC')

# Show the plot
plt.show()
plt.savefig('GRAPH_ACC_LIME_CIC.png')
plt.clf()

print('AUC - CIC LIME')

auc = np.trapz(y_axis_RF, x_axis_cic)/x_axis_cic[-1]
print('RF')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_DNN, x_axis_cic)/x_axis_cic[-1]
print('DNN')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_LGBM, x_axis_cic)/x_axis_cic[-1]
print('LGBM')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_SVM, x_axis_cic)/x_axis_cic[-1]
print('SVM')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_KNN, x_axis_cic)/x_axis_cic[-1]
print('KNN')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_ADA, x_axis_cic)/x_axis_cic[-1]
print('ADA')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_MLP, x_axis_cic)/x_axis_cic[-1]
print('MLP')
print(f"AUC: {auc}")


###########################################################################################################################################################################################################################################################################################################################
#SIMARGL
#LIME
    #RF
y_axis_RF =  [0.997, 0.996, 0.994, 0.997]
    #DNN
y_axis_DNN =  [0.776, 0.555, 0.555, 0.333]

y_axis_LGBM =   [0.999, 0.999, 0.994, 0.999]

y_axis_SVM = [0.932, 0.873, 0.913, 0.930]
    #ADA
y_axis_ADA =  [0.762, 0.804, 0.804, 0.670] 
    #KNN
y_axis_KNN = [0.998,0.997,0.988,  0.998   ]
    #MLP
y_axis_MLP =  [0.555, 0.555,0.555,0.555]

plt.clf()

# Plot the first line
plt.plot(x_axis_sml, y_axis_RF, label='RF', color='blue', linestyle='--', marker='o')

# Plot the second line
plt.plot(x_axis_sml, y_axis_DNN, label='DNN', color='red', linestyle='--', marker='x')

# Plot the third line
plt.plot(x_axis_sml, y_axis_LGBM, label='LGBM', color='green', linestyle='--', marker='s')

# Plot the fourth line
plt.plot(x_axis_sml, y_axis_SVM, label='SVM', color='purple', linestyle='--', marker='p')

# Plot the fifth line
plt.plot(x_axis_sml, y_axis_MLP, label='MLP', color='orange', linestyle='--', marker='h')

# Plot the sixth line
# plt.plot(x_axis_sml, y_axis_KNN, label='KNN', color='magenta', linestyle='--', marker='+')

# Plot the seventh line
plt.plot(x_axis_sml, y_axis_ADA, label='ADA', color='cyan', linestyle='--', marker='_')

# Enable grid lines (both major and minor grids)
plt.grid()

# Customize grid lines (optional)
# plt.grid()

# Add labels and a legend
plt.xlabel('Features')
plt.ylabel('Accuracy')
plt.legend()

# Set the title of the plot
# plt.title('Accuracy x Features - LIME SML')

# Show the plot
plt.show()
plt.savefig('GRAPH_ACC_LIME_SML.png')
plt.clf()

###########################################################################################################################################################################################################################################################################################################################

print('AUC - SML LIME')

auc = np.trapz(y_axis_RF, x_axis_sml)/x_axis_sml[-1]
print('RF')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_DNN, x_axis_sml)/x_axis_sml[-1]
print('DNN')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_LGBM, x_axis_sml)/x_axis_sml[-1]
print('LGBM')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_SVM, x_axis_sml)/x_axis_sml[-1]
print('SVM')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_KNN, x_axis_sml)/x_axis_sml[-1]
print('KNN')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_ADA, x_axis_sml)/x_axis_sml[-1]
print('ADA')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_MLP, x_axis_sml)/x_axis_sml[-1]
print('MLP')
print(f"AUC: {auc}")

