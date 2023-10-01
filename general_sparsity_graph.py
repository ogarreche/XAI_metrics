
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

x_axis = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] 

# CICIDS
#SHAP
#RF 
y_axis_RF =  [0.36363636363636365, 0.6883116883116883, 0.8311688311688312, 0.8831168831168831, 0.948051948051948, 0.961038961038961, 0.987012987012987, 0.987012987012987, 0.987012987012987, 0.987012987012987, 0.987012987012987] 
#DNN 
y_axis_DNN =  [0.18181818181818182, 0.8441558441558441, 0.922077922077922, 0.948051948051948, 0.961038961038961, 0.974025974025974, 0.974025974025974, 0.987012987012987, 0.987012987012987, 0.987012987012987, 0.987012987012987] 
#LGBM
y_axis_LGBM =  [0.2597402597402597, 0.8961038961038961, 0.961038961038961, 0.974025974025974, 0.987012987012987, 0.987012987012987, 0.987012987012987, 0.987012987012987, 0.987012987012987, 0.987012987012987, 0.987012987012987] 
#SVM 
y_axis_SVM =  [0.18181818181818182, 0.7922077922077922, 0.8701298701298701, 0.922077922077922, 0.948051948051948, 0.948051948051948, 0.961038961038961, 0.974025974025974, 0.974025974025974, 0.987012987012987, 0.987012987012987] 
#ADA
y_axis_ADA =  [0.18181818181818182, 0.961038961038961, 0.974025974025974, 0.987012987012987, 0.987012987012987, 0.987012987012987, 0.987012987012987, 0.987012987012987, 0.987012987012987, 0.987012987012987, 0.987012987012987] 
#KNN
y_axis_KNN = [0.19480519480519481, 0.6103896103896104, 0.6883116883116883, 0.8181818181818182, 0.8961038961038961, 0.935064935064935, 0.948051948051948, 0.948051948051948, 0.961038961038961, 0.974025974025974, 0.987012987012987] 
#MLP
y_axis_MLP =  [0.18181818181818182, 0.6103896103896104, 0.7662337662337663, 0.8831168831168831, 0.922077922077922, 0.935064935064935, 0.948051948051948, 0.948051948051948, 0.974025974025974, 0.987012987012987, 0.987012987012987] 



plt.clf()

# Plot the first line
plt.plot(x_axis, y_axis_RF, label='RF', color='blue', linestyle='--', marker='o')

# Plot the second line
plt.plot(x_axis, y_axis_DNN, label='DNN', color='red', linestyle='--', marker='x')

# Plot the third line
plt.plot(x_axis, y_axis_LGBM, label='LGBM', color='green', linestyle='--', marker='s')

# Plot the fourth line
plt.plot(x_axis, y_axis_SVM, label='SVM', color='purple', linestyle='--', marker='p')

# Plot the fifth line
plt.plot(x_axis, y_axis_MLP, label='MLP', color='orange', linestyle='--', marker='h')

# Plot the sixth line
plt.plot(x_axis, y_axis_KNN, label='KNN', color='magenta', linestyle='--', marker='+')

# Plot the seventh line
plt.plot(x_axis, y_axis_ADA, label='ADA', color='cyan', linestyle='--', marker='_')

# Enable grid lines (both major and minor grids)
plt.grid()

# Customize grid lines (optional)
# plt.grid()

# Add labels and a legend
plt.xlabel('Threshold')
plt.ylabel('Sparsity score')
plt.legend()

# Set the title of the plot
# plt.title('Sparsity SHAP CIC')

# Show the plot
plt.show()
plt.savefig('GRAPH_SPARSITY_SHAP_CIC.png')
plt.clf()

print('AUC - CICIDS SHAP')

auc = np.trapz(y_axis_RF, x_axis)
print('RF')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_DNN, x_axis)
print('DNN')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_LGBM, x_axis)
print('LGBM')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_SVM, x_axis)
print('SVM')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_KNN, x_axis)
print('KNN')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_ADA, x_axis)
print('ADA')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_MLP, x_axis)
print('MLP')
print(f"AUC: {auc}")


###########################################################################################################################################################################################################################################################################################################################
#SIMARGL
#SHAP
    #RF
y_axis_RF =  [0.2692307692307692, 0.5384615384615384, 0.7692307692307693, 0.8076923076923077, 0.8076923076923077, 0.8461538461538461, 0.8461538461538461, 0.8461538461538461, 0.8461538461538461, 0.9615384615384616, 0.9615384615384616] 
    #DNN
y_axis_DNN =  [0.15384615384615385, 0.6538461538461539, 0.6923076923076923, 0.8076923076923077, 0.8461538461538461, 0.8846153846153846, 0.8846153846153846, 0.8846153846153846, 0.9230769230769231, 0.9615384615384616, 0.9615384615384616] 
    #LGBM
y_axis_LGBM =  [0.19230769230769232, 0.8076923076923077, 0.8076923076923077, 0.8461538461538461, 0.8846153846153846, 0.8846153846153846, 0.8846153846153846, 0.8846153846153846, 0.9230769230769231, 0.9615384615384616, 0.9615384615384616] 
    #SVM
y_axis_SVM =  [0.15384615384615385, 0.5769230769230769, 0.6923076923076923, 0.8461538461538461, 0.8846153846153846, 0.9230769230769231, 0.9230769230769231, 0.9230769230769231, 0.9615384615384616, 0.9615384615384616, 0.9615384615384616] 
    #ADA
y_axis_ADA =  [0.15384615384615385, 0.5384615384615384, 0.6538461538461539, 0.8461538461538461, 0.8461538461538461, 0.8846153846153846, 0.8846153846153846, 0.9230769230769231, 0.9615384615384616, 0.9615384615384616, 0.9615384615384616] 
    #KNN
y_axis_KNN =  [0.2222222222222222, 0.6666666666666666, 0.8148148148148148, 0.8148148148148148, 0.8518518518518519, 0.8518518518518519, 0.8518518518518519, 0.8518518518518519, 0.8518518518518519, 0.8888888888888888, 0.9629629629629629] 

    #MLP
y_axis_MLP =  [0.5, 0.8461538461538461, 0.8846153846153846, 0.9230769230769231, 0.9230769230769231, 0.9615384615384616, 0.9615384615384616, 0.9615384615384616, 0.9615384615384616, 0.9615384615384616, 0.9615384615384616] 



plt.clf()

# Plot the first line
plt.plot(x_axis, y_axis_RF, label='RF', color='blue', linestyle='--', marker='o')

# Plot the second line
plt.plot(x_axis, y_axis_DNN, label='DNN', color='red', linestyle='--', marker='x')

# Plot the third line
plt.plot(x_axis, y_axis_LGBM, label='LGBM', color='green', linestyle='--', marker='s')

# Plot the fourth line
plt.plot(x_axis, y_axis_SVM, label='SVM', color='purple', linestyle='--', marker='p')

# Plot the fifth line
plt.plot(x_axis, y_axis_MLP, label='MLP', color='orange', linestyle='--', marker='h')

# Plot the sixth line
plt.plot(x_axis, y_axis_KNN, label='KNN', color='magenta', linestyle='--', marker='+')

# Plot the seventh line
plt.plot(x_axis, y_axis_ADA, label='ADA', color='cyan', linestyle='--', marker='_')

# Enable grid lines (both major and minor grids)
plt.grid()

# Customize grid lines (optional)
# plt.grid()

# Add labels and a legend
plt.xlabel('Threshold')
plt.ylabel('Sparsity score')
plt.legend()

# Set the title of the plot
# plt.title('Sparsity SHAP SML')

# Show the plot
plt.show()
plt.savefig('GRAPH_SPARSITY_SHAP_SML.png')
plt.clf()

print('AUC - SML SHAP')

auc = np.trapz(y_axis_RF, x_axis)
print('RF')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_DNN, x_axis)
print('DNN')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_LGBM, x_axis)
print('LGBM')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_SVM, x_axis)
print('SVM')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_KNN, x_axis)
print('KNN')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_ADA, x_axis)
print('ADA')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_MLP, x_axis)
print('MLP')
print(f"AUC: {auc}")

###########################################################################################################################################################################################################################################################################################################################
#CICIDS
#LIME
    #RF
y_axis_RF =  [0.012987012987012988, 0.19480519480519481, 0.4675324675324675, 0.7792207792207793, 0.922077922077922, 0.935064935064935, 0.961038961038961, 0.961038961038961, 0.974025974025974, 0.987012987012987, 0.987012987012987] 
    #DNN
y_axis_DNN =  [0.012987012987012988, 0.4935064935064935, 0.7792207792207793, 0.8831168831168831, 0.922077922077922, 0.948051948051948, 0.961038961038961, 0.987012987012987, 0.987012987012987, 0.987012987012987, 0.987012987012987] 
    #LGBM
y_axis_LGBM =  [0.012987012987012988, 0.05194805194805195, 0.09090909090909091, 0.15584415584415584, 0.33766233766233766, 0.7012987012987013, 0.8441558441558441, 0.9090909090909091, 0.948051948051948, 0.987012987012987, 0.987012987012987] 
    #SVM
y_axis_SVM =  [0.012987012987012988, 0.14285714285714285, 0.4025974025974026, 0.5584415584415584, 0.6623376623376623, 0.7922077922077922, 0.8701298701298701, 0.935064935064935, 0.961038961038961, 0.974025974025974, 0.987012987012987] 
    #ADA
y_axis_ADA =  [0.012987012987012988, 0.8311688311688312, 0.8961038961038961, 0.9090909090909091, 0.9090909090909091, 0.935064935064935, 0.935064935064935, 0.948051948051948, 0.974025974025974, 0.987012987012987, 0.987012987012987] 
    #KNN
y_axis_KNN =  [0.012987012987012988, 0.03896103896103896, 0.1038961038961039, 0.2077922077922078, 0.2987012987012987, 0.4025974025974026, 0.6883116883116883, 0.8831168831168831, 0.948051948051948, 0.987012987012987, 0.987012987012987] 
    #MLP
y_axis_MLP =  [0.012987012987012988, 0.03896103896103896, 0.14285714285714285, 0.3246753246753247, 0.6363636363636364, 0.8181818181818182, 0.922077922077922, 0.948051948051948, 0.974025974025974, 0.987012987012987, 0.987012987012987] 


plt.clf()

# Plot the first line
plt.plot(x_axis, y_axis_RF, label='RF', color='blue', linestyle='--', marker='o')

# Plot the second line
plt.plot(x_axis, y_axis_DNN, label='DNN', color='red', linestyle='--', marker='x')

# Plot the third line
plt.plot(x_axis, y_axis_LGBM, label='LGBM', color='green', linestyle='--', marker='s')

# Plot the fourth line
plt.plot(x_axis, y_axis_SVM, label='SVM', color='purple', linestyle='--', marker='p')

# Plot the fifth line
plt.plot(x_axis, y_axis_MLP, label='MLP', color='orange', linestyle='--', marker='h')

# Plot the sixth line
plt.plot(x_axis, y_axis_KNN, label='KNN', color='magenta', linestyle='--', marker='+')

# Plot the seventh line
plt.plot(x_axis, y_axis_ADA, label='ADA', color='cyan', linestyle='--', marker='_')

# Enable grid lines (both major and minor grids)
plt.grid()

# Customize grid lines (optional)
# plt.grid()

# Add labels and a legend
plt.xlabel('Threshold')
plt.ylabel('Sparsity score')
plt.legend()

# Set the title of the plot
# plt.title('Sparsity LIME CIC')

# Show the plot
plt.show()
plt.savefig('GRAPH_SPARSITY_LIME_CIC.png')
plt.clf()

print('AUC - CICIDS LIME')

auc = np.trapz(y_axis_RF, x_axis)
print('RF')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_DNN, x_axis)
print('DNN')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_LGBM, x_axis)
print('LGBM')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_SVM, x_axis)
print('SVM')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_KNN, x_axis)
print('KNN')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_ADA, x_axis)
print('ADA')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_MLP, x_axis)
print('MLP')
print(f"AUC: {auc}")

###########################################################################################################################################################################################################################################################################################################################
#SIMARGL
#LIME
    #RF
y_axis_RF =  [0.038461538461538464, 0.038461538461538464, 0.07692307692307693, 0.15384615384615385, 0.3076923076923077, 0.5769230769230769, 0.7692307692307693, 0.8846153846153846, 0.9230769230769231, 0.9615384615384616, 0.9615384615384616] 

    #DNN
y_axis_DNN =  [0.038461538461538464, 0.15384615384615385, 0.5, 0.6153846153846154, 0.7692307692307693, 0.8461538461, 0.8461538461538461, 0.9230769230769231, 0.9615384615384616, 0.9615384615384616, 0.9615384615384616] 
#check this one again!                                             
    #LGBM
y_axis_LGBM =  [0.038461538461538464, 0.038461538461538464, 0.07692307692307693, 0.19230769230769232, 0.2692307692307692, 0.34615384615384615, 0.6153846153846154, 0.8846153846153846, 0.9230769230769231, 0.9615384615384616, 0.9615384615384616] 

    #SVM
y_axis_SVM =  [0.038461538461538464, 0.11538461538461539, 0.2692307692307692, 0.5384615384615384, 0.6923076923076923, 0.6923076923076923, 0.7692307692307693, 0.8461538461538461, 0.8461538461538461, 0.9230769230769231, 0.9615384615384616] 

    #ADA
y_axis_ADA =  [0.038461538461538464, 0.07692307692307693, 0.4230769230769231, 0.6153846153846154, 0.7692307692307693, 0.7692307692307693, 0.8076923076923077, 0.8076923076923077, 0.8461538461538461, 0.9230769230769231, 0.9615384615384616] 

    #KNN
y_axis_KNN =  [0.038461538461538464, 0.07692307692307693, 0.23076923076923078, 0.34615384615384615, 0.5384615384615384, 0.5769230769230769, 0.7692307692307693, 0.7692307692307693, 0.8461538461538461, 0.8846153846153846, 0.9615384615384616] 
    #MLP
y_axis_MLP =  [0.038461538461538464, 0.038461538461538464, 0.11538461538461539, 0.19230769230769232, 0.38461538461538464, 0.5769230769230769, 0.7692307692307693, 0.8461538461538461, 0.9615384615384616, 0.9615384615384616, 0.9615384615384616] 



plt.clf()

# Plot the first line
plt.plot(x_axis, y_axis_RF, label='RF', color='blue', linestyle='--', marker='o')

# Plot the second line
plt.plot(x_axis, y_axis_DNN, label='DNN', color='red', linestyle='--', marker='x')

# Plot the third line
plt.plot(x_axis, y_axis_LGBM, label='LGBM', color='green', linestyle='--', marker='s')

# Plot the fourth line
plt.plot(x_axis, y_axis_SVM, label='SVM', color='purple', linestyle='--', marker='p')

# Plot the fifth line
plt.plot(x_axis, y_axis_MLP, label='MLP', color='orange', linestyle='--', marker='h')

# Plot the sixth line
plt.plot(x_axis, y_axis_KNN, label='KNN', color='magenta', linestyle='--', marker='+')

# Plot the seventh line
plt.plot(x_axis, y_axis_ADA, label='ADA', color='cyan', linestyle='--', marker='_')

# Enable grid lines (both major and minor grids)
plt.grid()

# Customize grid lines (optional)
# plt.grid()

# Add labels and a legend
plt.xlabel('Threshold')
plt.ylabel('Sparsity score')
plt.legend()

# Set the title of the plot
# plt.title('Sparsity LIME SML')

# Show the plot
plt.show()
plt.savefig('GRAPH_SPARSITY_LIME_SML.png')
plt.clf()

print('AUC - SML LIME')

auc = np.trapz(y_axis_RF, x_axis)
print('RF')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_DNN, x_axis)
print('DNN')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_LGBM, x_axis)
print('LGBM')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_SVM, x_axis)
print('SVM')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_KNN, x_axis)
print('KNN')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_ADA, x_axis)
print('ADA')
print(f"AUC: {auc}")

auc = np.trapz(y_axis_MLP, x_axis)
print('MLP')
print(f"AUC: {auc}")

###########################################################################################################################################################################################################################################################################################################################
