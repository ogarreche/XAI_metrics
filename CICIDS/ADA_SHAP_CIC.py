

###################################################
#               Parameter Setting                #
###################################################

fraction= 0.5 # how much of that database you want to use
frac_normal = .2 #how much of the normal classification you want to reduce
split = 0.70 # how you want to split the train/test data (this is percentage fro train)

#Model Parameters

n_estimators=100
learning_rate=0.5

# XAI Samples
samples = 500

# Specify the name of the output text file
output_file_name = "ADA_SHAP_CIC.txt"
with open(output_file_name, "w") as f: print('---------------------------------------------------------------------------------', file = f)
###################################################
###################################################
###################################################
###################################################
print('---------------------------------------------------------------------------------')
print('Initializing ADA program')
print('---------------------------------------------------------------------------------')
print('')
#Importing libraries
#----------------------------------------------------------------------------------------------------------
print('---------------------------------------------------------------------------------')
print('Importing Libraries')
print('---------------------------------------------------------------------------------')
print('')

import pandas as pd
#Loading numpy
import numpy as np
# Setting random seed
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.multiclass import OneVsRestClassifier
from collections import Counter
from sklearn.preprocessing import label_binarize
import time

np.random.seed(0)

from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import shap
import sklearn
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import lime
#----------------------------------------------------------------------------------------------------------
#Selecting features from db
print('Defining features of interest')
print('---------------------------------------------------------------------------------')
print('')
'''
########################################### CICIDS Features ########################################
'''

# Select which feature method you want to use by uncommenting it.

'''
all features
'''


req_cols = [' Destination Port',' Flow Duration',' Total Fwd Packets',' Total Backward Packets','Total Length of Fwd Packets',' Total Length of Bwd Packets',' Fwd Packet Length Max',' Fwd Packet Length Min',' Fwd Packet Length Mean',' Fwd Packet Length Std','Bwd Packet Length Max',' Bwd Packet Length Min',' Bwd Packet Length Mean',' Bwd Packet Length Std','Flow Bytes/s',' Flow Packets/s',' Flow IAT Mean',' Flow IAT Std',' Flow IAT Max',' Flow IAT Min','Fwd IAT Total',' Fwd IAT Mean',' Fwd IAT Std',' Fwd IAT Max',' Fwd IAT Min','Bwd IAT Total',' Bwd IAT Mean',' Bwd IAT Std',' Bwd IAT Max',' Bwd IAT Min','Fwd PSH Flags',' Bwd PSH Flags',' Fwd URG Flags',' Bwd URG Flags',' Fwd Header Length',' Bwd Header Length','Fwd Packets/s',' Bwd Packets/s',' Min Packet Length',' Max Packet Length',' Packet Length Mean',' Packet Length Std',' Packet Length Variance','FIN Flag Count',' SYN Flag Count',' RST Flag Count',' PSH Flag Count',' ACK Flag Count',' URG Flag Count',' CWE Flag Count',' ECE Flag Count',' Down/Up Ratio',' Average Packet Size',' Avg Fwd Segment Size',' Avg Bwd Segment Size',' Fwd Header Length','Fwd Avg Bytes/Bulk',' Fwd Avg Packets/Bulk',' Fwd Avg Bulk Rate',' Bwd Avg Bytes/Bulk',' Bwd Avg Packets/Bulk','Bwd Avg Bulk Rate','Subflow Fwd Packets',' Subflow Fwd Bytes',' Subflow Bwd Packets',' Subflow Bwd Bytes','Init_Win_bytes_forward',' Init_Win_bytes_backward',' act_data_pkt_fwd',' min_seg_size_forward','Active Mean',' Active Std',' Active Max',' Active Min','Idle Mean',' Idle Std',' Idle Max',' Idle Min',' Label']


#----------------------------------------------------------------------------------------------------------
#Defining metric functions
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
     # plt.plot(fpr[i], tpr[i], color='darkorange', lw=2)
      #print('AUC for Class {}: {}'.format(i+1, auc(fpr[i], tpr[i])))
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




#----------------------------------------------------------------------------------------------------------
#Loading Database
print('Loading Databases')
print('---------------------------------------------------------------------------------')
print('')

fraction =fraction
df0 = pd.read_csv ('cicids_db/Wednesday-workingHours.pcap_ISCX.csv', usecols=req_cols).sample(frac = fraction)

df1 = pd.read_csv ('cicids_db/Tuesday-WorkingHours.pcap_ISCX.csv', usecols=req_cols).sample(frac = fraction)


df2 = pd.read_csv ('cicids_db/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv', usecols=req_cols).sample(frac = fraction)


df3 = pd.read_csv ('cicids_db/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv', usecols=req_cols).sample(frac = fraction)


df4 = pd.read_csv ('cicids_db/Monday-WorkingHours.pcap_ISCX.csv', usecols=req_cols).sample(frac = fraction)


df5 = pd.read_csv ('cicids_db/Friday-WorkingHours-Morning.pcap_ISCX.csv', usecols=req_cols).sample(frac = fraction)


df6 = pd.read_csv ('cicids_db/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv', usecols=req_cols).sample(frac = fraction)


df7 = pd.read_csv ('cicids_db/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv', usecols=req_cols).sample(frac = fraction)



frames = [df0, df1,  df2, df3, df4, df5,df6, df7]

df = pd.concat(frames,ignore_index=True)



print('---------------------------------------------------------------------------------')
print('Removing top features for acc')
print('---------------------------------------------------------------------------------')

# df.pop(' Destination Port')
# df.pop(' min_seg_size_forward')
# df.pop(' Min Packet Length')
# df.pop(' Bwd Packet Length Min')
# df.pop(' Fwd Packet Length Std')
# df.pop(' Subflow Fwd Bytes')
# df.pop(' Avg Fwd Segment Size')
# df.pop('Total Length of Fwd Packets')
# df.pop(' Init_Win_bytes_backward')
# df.pop(' PSH Flag Count')
# df.pop('Fwd IAT Total')
# df.pop(' Flow Duration')
# df.pop(' Fwd Packet Length Mean')
# df.pop('Bwd Packet Length Max')
# df.pop(' Bwd Packets/s')
# df.pop(' Average Packet Size')
# df.pop(' ACK Flag Count')
# df.pop(' Subflow Bwd Packets')
# df.pop('Fwd Packets/s')
# df.pop(' URG Flag Count')
# df.pop('Init_Win_bytes_forward')
# df.pop(' Flow IAT Std')
# df.pop('Fwd PSH Flags')
# df.pop(' Packet Length Mean')
# df.pop(' Active Min')
# df.pop(' Total Length of Bwd Packets')
# df.pop(' Total Backward Packets')
# df.pop(' Fwd IAT Mean')
# df.pop(' Idle Max')
# df.pop('Subflow Fwd Packets')
# df.pop(' Flow IAT Mean')
# df.pop(' Active Max')
# df.pop('Idle Mean')
# df.pop(' Flow IAT Min')
# df.pop(' Bwd Packet Length Mean')
# df.pop(' Fwd Packet Length Max')
# df.pop(' Fwd Packet Length Min')
# df.pop(' Fwd IAT Max')
# df.pop(' Packet Length Variance')
# df.pop(' Packet Length Std')
# df.pop(' Bwd IAT Std')
# df.pop('Active Mean')
# df.pop(' Fwd IAT Min')
# df.pop(' Bwd Header Length')
# df.pop('Bwd IAT Total')
# df.pop(' Avg Bwd Segment Size')
# df.pop(' Max Packet Length')
# df.pop(' Fwd IAT Std')
# df.pop(' Idle Min')
# df.pop(' Bwd IAT Max')
# df.pop(' Bwd Packet Length Std')
# df.pop(' SYN Flag Count')
# df.pop(' Bwd IAT Min')
# df.pop('FIN Flag Count')
# df.pop(' Down/Up Ratio')
# df.pop(' Idle Std')
# df.pop(' Flow IAT Max')
# df.pop(' Bwd IAT Mean')
# df.pop(' Total Fwd Packets')
# df.pop(' Fwd Header Length')
# df.pop(' Subflow Bwd Bytes')
# df.pop(' Active Std')
# df.pop(' act_data_pkt_fwd')
# df.pop(' Fwd URG Flags')
# df.pop(' Bwd URG Flags')
# df.pop(' Bwd PSH Flags')
# df.pop(' CWE Flag Count')
# df.pop(' ECE Flag Count')
# df.pop('Flow Bytes/s')
# df.pop('Bwd Avg Bulk Rate')
# df.pop(' RST Flag Count')
# df.pop(' Bwd Avg Bytes/Bulk')
# df.pop(' Fwd Avg Bulk Rate')
# df.pop(' Fwd Avg Packets/Bulk')
# df.pop('Fwd Avg Bytes/Bulk')
# df.pop(' Flow Packets/s')
# df.pop(' Bwd Avg Packets/Bulk')



#---------------------------------------------------------------------
y = df.pop(' Label')
df = df.assign(Label = y)



print('---------------------------------------------------------------------------------')
print('Reducing Normal rows')
print('---------------------------------------------------------------------------------')
print('')


#filters

filtered_normal = df[df['Label'] == 'BENIGN']

#reduce

reduced_normal = filtered_normal.sample(frac=frac_normal)

#join

df = pd.concat([df[df['Label'] != 'BENIGN'], reduced_normal])

''' ---------------------------------------------------------------'''
df_max_scaled = df.copy()


# y = df_max_scaled['Label'].replace({'DoS GoldenEye': 'Dos/Ddos', 'DoS Hulk': 'Dos/Ddos', 'DoS Slowhttptest': 'Dos/Ddos', 'DoS slowloris': 'Dos/Ddos', 'Heartbleed': 'Dos/Ddos', 'DDoS': 'Dos/Ddos','FTP-Patator': 'Brute Force', 'SSH-Patator': 'Brute Force','Web Attack - Brute Force': 'Web Attack', 'Web Attack - Sql Injection': 'Web Attack', 'Web Attack - XSS': 'Web Attack'})
y = df_max_scaled['Label'].replace({'DoS GoldenEye': 'Dos/Ddos',
'DoS Hulk': 'Dos/Ddos',
'DoS Slowhttptest': 'Dos/Ddos',
'DoS slowloris': 'Dos/Ddos',
'Heartbleed': 'Dos/Ddos',
'DDoS': 'Dos/Ddos',
'FTP-Patator': 'Brute Force',
'SSH-Patator': 'Brute Force',
'Web Attack - Brute Force': 'Web Attack',
'Web Attack - Sql Injection': 'Web Attack',
'Web Attack - XSS': 'Web Attack',
'Web Attack XSS': 'Web Attack',
'Web Attack Sql Injection': 'Web Attack',
'Web Attack Brute Force': 'Web Attack'
})
df_max_scaled.pop('Label')



print('---------------------------------------------------------------------------------')
print('---------------------------------------------------------------------------------')
print('Normalizing')
print('---------------------------------------------------------------------------------')

for col in df_max_scaled.columns:
    t = abs(df_max_scaled[col].max())
    df_max_scaled[col] = df_max_scaled[col]/t
df_max_scaled
df = df_max_scaled.assign( Label = y)
df = df.fillna(0)

print('---------------------------------------------------------------------------------')
print('Balance Datasets')
print('---------------------------------------------------------------------------------')
print('')

counter = Counter(y)
print(counter)

# call balance operation until all labels have the same size
counter_list = list(counter.values())
for i in range(1,len(counter_list)):
    if counter_list[i-1] != counter_list[i]:
        df, y = oversample(df, y)

counter = Counter(y)


df = df.assign(Label = y)
print('train len',counter)

y = df.pop('Label')
X = df

df = df.assign(Label = y)


print('---------------------------------------------------------------------------------')
print('Spliting Train and Test')
print('---------------------------------------------------------------------------------')


X_train,X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, train_size=split)
df = X.assign( Label = y)


# Separate features and labels
#y_train, label = pd.factorize(y_train)
#X_train.pop('Label')
#----------------------------------------------------------------------------------------------------------
#Model Construction
print('---------------------------------------------------------------------------------')
print('Model training')
print('---------------------------------------------------------------------------------')

abc = AdaBoostClassifier(n_estimators=n_estimators,learning_rate=learning_rate)
#----------------------------------------------------------------------------------------------------------
#Running the model

#START TIMER MODEL
start = time.time()
model = abc.fit(X_train, y_train)
#END TIMER MODEL
end = time.time()
print('ELAPSE TIME MODEL: ',(end - start)/60, 'min')

#----------------------------------------------------------------------------------------------------------
#Data preprocessing
#X_test = np.array(test[features])

#----------------------------------------------------------------------------------------------------------
# Model predictions 

print('---------------------------------------------------------------------------------')
print('Model Prediction')
print('---------------------------------------------------------------------------------')

#START TIMER PREDICTION
start = time.time()

y_pred = model.predict(X_test)

#END TIMER PREDICTION
end = time.time()
print('ELAPSE TIME PREDICTION: ',(end - start)/60, 'min')

#----------------------------------------------------------------------------------------------------------

u, label = pd.factorize(y_test)

#pred_label = label[y_pred]
pred_label = y_pred
#----------------------------------------------------------------------------------------------------------
# Confusion Matrix
print('---------------------------------------------------------------------------------')
print('Generating Confusion Matrix')
print('---------------------------------------------------------------------------------')
print('')

confusion_matrix = pd.crosstab(y_test, pred_label,rownames=['Actual ALERT'],colnames = ['Predicted ALERT'], dropna=False).sort_index(axis=0).sort_index(axis=1)
all_unique_values = sorted(set(pred_label) | set(y_test))
z = np.zeros((len(all_unique_values), len(all_unique_values)))
rows, cols = confusion_matrix.shape
z[:rows, :cols] = confusion_matrix
confusion_matrix  = pd.DataFrame(z, columns=all_unique_values, index=all_unique_values)
with open(output_file_name, "a") as f: print(confusion_matrix, file = f)

#True positives and False positives and negatives
FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
TP = np.diag(confusion_matrix)
TN = confusion_matrix.values.sum() - (FP + FN + TP)
#Sum each Labels TP,TN,FP,FN in one overall measure
TP_total = sum(TP)
TN_total = sum(TN)
FP_total = sum(FP)
FN_total = sum(FN)

#data preprocessin because numbers are getting big
TP_total = np.array(TP_total,dtype=np.float64)
TN_total = np.array(TN_total,dtype=np.float64)
FP_total = np.array(FP_total,dtype=np.float64)
FN_total = np.array(FN_total,dtype=np.float64)

#----------------------------------------------------------------------------------------------------------
#Metrics measure overall
Acc = ACC(TP_total,TN_total, FP_total, FN_total)
Precision = PRECISION(TP_total, FP_total)
Recall = RECALL(TP_total, FN_total)
F1 = F1(Recall,Precision)
BACC = BACC(TP_total,TN_total, FP_total, FN_total)
MCC = MCC(TP_total,TN_total, FP_total, FN_total)
with open(output_file_name, "a") as f: print('Accuracy: ', Acc, file = f)
print('Precision: ', Precision )
print('Recall: ', Recall )
print('F1: ', F1 )
print('BACC: ', BACC)
print('MCC: ', MCC)

for i in range(0,len(TP)):
   # Acc_2 = ACC_2(TP[i],FN[i])
    Acc = ACC(TP[i],TN[i], FP[i], FN[i])
    print('Accuracy: ', label[i] ,' - ' , Acc)
print('---------------------------------------------------------------------------------')



#----------------------------------------
y_score = abc.predict_proba(X_test)
y_test_bin = label_binarize(y_test,classes = [0,1,2,3,4,5,6])
n_classes = y_test_bin.shape[1]
try:
    print('rocauc is ',roc_auc_score(y_test_bin,y_score, multi_class='iovr'))
except:
    print('rocauc is nan')

# print('---------------------------------------------------------------------------------')
# print('Generating SHAP explanation')
# print('---------------------------------------------------------------------------------')
# print('')


# print('---------------------------------------------------------------------------------')
# print('Generating explainer')
# print('---------------------------------------------------------------------------------')
# print('')
# test = X_test
# train = X_train
# # ## Summary Bar Plot Global
# start_index = 0
# end_index = samples
# # test.pop('Label')
# # test.pop('is_train')
# # print(label2)
# explainer = shap.KernelExplainer(abc.predict_proba, test[start_index:end_index])

# shap_values = explainer.shap_values(test[start_index:end_index])

# # shap.summary_plot(shap_values = shap_values,
# #                   features = test[start_index:end_index],class_names=[label[0],label[1],label[2],label[3],label[4],label[5],label[6]],show=False)
# # plt.savefig('ADA_Shap_Summary_global.png')
# plt.clf()


# vals= np.abs(shap_values).mean(1)
# feature_importance = pd.DataFrame(list(zip(train.columns, sum(vals))), columns=['col_name','feature_importance_vals'])
# feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
# feature_importance.head()
# print(feature_importance.to_string())



# print('---------------------------------------------------------------------------------')
# # feature_importance_vals = 'feature_importance_vals'  # Replace with the name of the column you want to extract
# feature_val = feature_importance['feature_importance_vals'].tolist()

# # col_name = 'col_name'  # Replace with the name of the column you want to extract
# feature_name = feature_importance['col_name'].tolist()


# # for item1, item2 in zip(feature_name, feature_val):
# #     print(item1, item2)


# # Use zip to combine the two lists, sort based on list1, and then unzip them
# zipped_lists = list(zip(feature_name, feature_val))
# zipped_lists.sort(key=lambda x: x[1],reverse=True)

# # Convert the sorted result back into separate lists
# sorted_list1, sorted_list2 = [list(x) for x in zip(*zipped_lists)]

# for k in sorted_list1:
#   with open(output_file_name, "a") as f: print("df.pop('",k,"')", sep='', file = f)

# print('---------------------------------------------------------------------------------')



# print('---------------------------------------------------------------------------------')
# print('Generating Sparsity Graph')
# print('---------------------------------------------------------------------------------')
# print('')

# # Find the minimum and maximum values in the list
# min_value = min(feature_val)
# max_value = max(feature_val)

# # Normalize the list to the range [0, 1]
# try:
#     normalized_list = [(x - min_value) / (max_value - min_value) for x in feature_val]
# except:
#     normalized_list = [ 0 for x in feature_val] 

# # print(feature_name,normalized_list,'\n')
# # for item1, item2 in zip(feature_name, normalized_list):
# #     print(item1, item2)

# #calculating Sparsity

# # Define the threshold
# threshold = 1e-10

# # Initialize a count variable to keep track of values below the threshold
# count_below_threshold = 0

# # Iterate through the list and count values below the threshold
# for value in normalized_list:
#     if value < threshold:
#         count_below_threshold += 1

# Sparsity = count_below_threshold/len(normalized_list)
# Spar = []
# print('Sparsity = ',Sparsity)
# X_axis = []
# #----------------------------------------------------------------------------
# for i in range(0, 11):
#     i/10
#     threshold = i/10
#     for value in normalized_list:
#         if value < threshold:
#             count_below_threshold += 1

#     Sparsity = count_below_threshold/len(normalized_list)
#     Spar.append(Sparsity)
#     X_axis.append(i/10)
#     count_below_threshold = 0


# #---------------------------------------------------------------------------

# with open(output_file_name, "a") as f: print('y_axis_RF = ', Spar ,'', file = f)
# with open(output_file_name, "a") as f:print('x_axis_RF = ', X_axis ,'', file = f)

# plt.clf()

# # Create a plot
# plt.plot(X_axis, Spar, marker='o', linestyle='-')

# # Set labels for the axes
# plt.xlabel('X-Axis')
# plt.ylabel('Y-Axis')

# # Set the title of the plot
# plt.title('Values vs. X-Axis')

# # Show the plot
# # plt.show()
# plt.savefig('sparsity.png')
# plt.clf()
