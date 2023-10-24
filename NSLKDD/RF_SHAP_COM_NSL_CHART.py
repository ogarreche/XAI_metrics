#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing required libraries
import numpy as np
import pandas as pd
import pickle # saving and loading trained model
from os import path

# importing required libraries for normalizing data
from sklearn import preprocessing
from sklearn.preprocessing import (StandardScaler, OrdinalEncoder,LabelEncoder, MinMaxScaler, OneHotEncoder)
from sklearn.preprocessing import Normalizer, MaxAbsScaler , RobustScaler, PowerTransformer

# importing library for plotting/

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.metrics import accuracy_score # for calculating accuracy of model
from sklearn.model_selection import train_test_split # for splitting the dataset for training and testing
from sklearn.metrics import classification_report # for generating a classification report of model

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from keras.layers import Dense # importing dense layer

from keras.layers import Input
from keras.models import Model
# representation of model layers
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix
import shap
from imblearn.over_sampling import SMOTE
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer


# In[ ]:





# In[2]:


#Defining metric functions
def ACC(TP,TN,FP,FN):
    Acc = (TP+TN)/(TP+FP+FN+TN)
    return Acc
def ACC_2 (TP, FN):
    ac = (TP/(TP+FN))
    return ac
def PRECISION(TP,FP):
    eps = 1e-7
    Precision = TP/(TP+FP+eps)
    

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
    eps = 1e-7
    MCC = (TN*TP-FN*FP)/(((TP+FP+eps)*(TP+FN+eps)*(TN+FP+eps)*(TN+FN+eps))**.5)
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


# In[3]:


# attach the column names to the dataset
feature=["duration","protocol_type","service","flag","src_bytes","dst_bytes","land","wrong_fragment","urgent","hot",
          "num_failed_logins","logged_in","num_compromised","root_shell","su_attempted","num_root","num_file_creations","num_shells",
          "num_access_files","num_outbound_cmds","is_host_login","is_guest_login","count","srv_count","serror_rate","srv_serror_rate",
          "rerror_rate","srv_rerror_rate","same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count", 
          "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate","dst_host_srv_diff_host_rate","dst_host_serror_rate",
          "dst_host_srv_serror_rate","dst_host_rerror_rate","dst_host_srv_rerror_rate","label","difficulty"]
# KDDTrain+_2.csv & KDDTest+_2.csv are the datafiles without the last column about the difficulty score
# these have already been removed.

train='KDDTrain+.txt'
test='KDDTest+.txt'

df=pd.read_csv(train,names=feature)
df_test=pd.read_csv(test,names=feature)

# shape, this gives the dimensions of the dataset
print('Dimensions of the Training set:',df.shape)
print('Dimensions of the Test set:',df_test.shape)


# In[4]:


df.drop(['difficulty'],axis=1,inplace=True)
df_test.drop(['difficulty'],axis=1,inplace=True)


# In[5]:


print('Label distribution Training set:')
print(df['label'].value_counts())
print()
print('Label distribution Test set:')
print(df_test['label'].value_counts())


# In[6]:


# colums that are categorical and not binary yet: protocol_type (column 2), service (column 3), flag (column 4).
# explore categorical features
print('Training set:')
for col_name in df.columns:
    if df[col_name].dtypes == 'object' :
        unique_cat = len(df[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))

#see how distributed the feature service is, it is evenly distributed and therefore we need to make dummies for all.
print()
print('Distribution of categories in service:')
print(df['service'].value_counts().sort_values(ascending=False).head())


# In[7]:


# Test set
print('Test set:')
for col_name in df_test.columns:
    if df_test[col_name].dtypes == 'object' :
        unique_cat = len(df_test[col_name].unique())
        print("Feature '{col_name}' has {unique_cat} categories".format(col_name=col_name, unique_cat=unique_cat))


# In[8]:


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
categorical_columns=['protocol_type', 'service', 'flag']
# insert code to get a list of categorical columns into a variable, categorical_columns
categorical_columns=['protocol_type', 'service', 'flag'] 
 # Get the categorical values into a 2D numpy array
df_categorical_values = df[categorical_columns]
testdf_categorical_values = df_test[categorical_columns]
df_categorical_values.head()


# In[9]:


# protocol type
unique_protocol=sorted(df.protocol_type.unique())
string1 = 'Protocol_type_'
unique_protocol2=[string1 + x for x in unique_protocol]
# service
unique_service=sorted(df.service.unique())
string2 = 'service_'
unique_service2=[string2 + x for x in unique_service]
# flag
unique_flag=sorted(df.flag.unique())
string3 = 'flag_'
unique_flag2=[string3 + x for x in unique_flag]
# put together
dumcols=unique_protocol2 + unique_service2 + unique_flag2
print(dumcols)

#do same for test set
unique_service_test=sorted(df_test.service.unique())
unique_service2_test=[string2 + x for x in unique_service_test]
testdumcols=unique_protocol2 + unique_service2_test + unique_flag2


# In[10]:


df_categorical_values_enc=df_categorical_values.apply(LabelEncoder().fit_transform)
print(df_categorical_values_enc.head())
# test set
testdf_categorical_values_enc=testdf_categorical_values.apply(LabelEncoder().fit_transform)


# In[11]:


enc = OneHotEncoder()
df_categorical_values_encenc = enc.fit_transform(df_categorical_values_enc)
df_cat_data = pd.DataFrame(df_categorical_values_encenc.toarray(),columns=dumcols)
# test set
testdf_categorical_values_encenc = enc.fit_transform(testdf_categorical_values_enc)
testdf_cat_data = pd.DataFrame(testdf_categorical_values_encenc.toarray(),columns=testdumcols)

df_cat_data.head()


# In[12]:


trainservice=df['service'].tolist()
testservice= df_test['service'].tolist()
difference=list(set(trainservice) - set(testservice))
string = 'service_'
difference=[string + x for x in difference]
difference


# In[13]:


for col in difference:
    testdf_cat_data[col] = 0

testdf_cat_data.shape


# In[14]:


newdf=df.join(df_cat_data)
newdf.drop('flag', axis=1, inplace=True)
newdf.drop('protocol_type', axis=1, inplace=True)
newdf.drop('service', axis=1, inplace=True)
# test data
newdf_test=df_test.join(testdf_cat_data)
newdf_test.drop('flag', axis=1, inplace=True)
newdf_test.drop('protocol_type', axis=1, inplace=True)
newdf_test.drop('service', axis=1, inplace=True)
print(newdf.shape)
print(newdf_test.shape)


# In[15]:


# take label column
labeldf=newdf['label']
labeldf_test=newdf_test['label']
# change the label column
newlabeldf=labeldf.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                           ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                           'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})
newlabeldf_test=labeldf_test.replace({ 'normal' : 0, 'neptune' : 1 ,'back': 1, 'land': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1, 'worm': 1,
                           'ipsweep' : 2,'nmap' : 2,'portsweep' : 2,'satan' : 2,'mscan' : 2,'saint' : 2
                           ,'ftp_write': 3,'guess_passwd': 3,'imap': 3,'multihop': 3,'phf': 3,'spy': 3,'warezclient': 3,'warezmaster': 3,'sendmail': 3,'named': 3,'snmpgetattack': 3,'snmpguess': 3,'xlock': 3,'xsnoop': 3,'httptunnel': 3,
                           'buffer_overflow': 4,'loadmodule': 4,'perl': 4,'rootkit': 4,'ps': 4,'sqlattack': 4,'xterm': 4})
# put the new label column back
newdf['label'] = newlabeldf
newdf_test['label'] = newlabeldf_test
print(newdf['label'].head())


# In[16]:


# creating a dataframe with multi-class labels (Dos,Probe,R2L,U2R,normal)
multi_data = newdf.copy()
multi_label = pd.DataFrame(multi_data.label)

multi_data_test=newdf_test.copy()
multi_label_test = pd.DataFrame(multi_data_test.label)


# In[17]:


# using standard scaler for normalizing
std_scaler = StandardScaler()
def standardization(df,col):
    for i in col:
        arr = df[i]
        arr = np.array(arr)
        df[i] = std_scaler.fit_transform(arr.reshape(len(arr),1))
    return df

numeric_col = multi_data.select_dtypes(include='number').columns
data = standardization(multi_data,numeric_col)
numeric_col_test = multi_data_test.select_dtypes(include='number').columns
data_test = standardization(multi_data_test,numeric_col_test)


# In[18]:


# label encoding (0,1,2,3,4) multi-class labels (Dos,normal,Probe,R2L,U2R)
le2 = preprocessing.LabelEncoder()
le2_test = preprocessing.LabelEncoder()
enc_label = multi_label.apply(le2.fit_transform)
enc_label_test = multi_label_test.apply(le2_test.fit_transform)
multi_data = multi_data.copy()
multi_data_test = multi_data_test.copy()

multi_data['intrusion'] = enc_label
multi_data_test['intrusion'] = enc_label_test

#y_mul = multi_data['intrusion']
multi_data
multi_data_test


# In[19]:


multi_data.drop(labels= [ 'label'], axis=1, inplace=True)
multi_data
multi_data_test.drop(labels= [ 'label'], axis=1, inplace=True)
multi_data_test


# In[20]:


y_train_multi= multi_data[['intrusion']]
X_train_multi= multi_data.drop(labels=['intrusion'], axis=1)

print('X_train has shape:',X_train_multi.shape,'\ny_train has shape:',y_train_multi.shape)

y_test_multi= multi_data_test[['intrusion']]
X_test_multi= multi_data_test.drop(labels=['intrusion'], axis=1)

print('X_test has shape:',X_test_multi.shape,'\ny_test has shape:',y_test_multi.shape)


# In[ ]:





# In[ ]:





# In[21]:


from sklearn.preprocessing import LabelBinarizer

y_train_multi = LabelBinarizer().fit_transform(y_train_multi)
y_train_multi

y_test_multi = LabelBinarizer().fit_transform(y_test_multi)
y_test_multi


# In[ ]:





# In[22]:


Y_train=y_train_multi.copy()
X_train=X_train_multi.copy()

Y_test=y_test_multi.copy()
X_test=X_test_multi.copy()


# In[23]:


from sklearn.preprocessing import MinMaxScaler
X_test = X_test[X_train.columns]
# Create a MinMaxScaler object
scaler = MinMaxScaler()

# Fit the scaler to the training data and transform it in one step
X_train_n = scaler.fit_transform(X_train)

# Transform the test data using the fitted scaler
X_test_n = scaler.transform(X_test)


# Print the transformed data
print(X_train)
print(X_test)


# In[24]:


import pandas as pd

# Convert normalized data back to DataFrame
X_train = pd.DataFrame(X_train_n, columns=X_train.columns)
X_test = pd.DataFrame(X_test_n, columns=X_test.columns)


# In[106]:


# Import necessary libraries
from sklearn.ensemble import RandomForestClassifier
import time

# Create a RandomForestClassifier instance
rf = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5,
                            min_samples_leaf=3, class_weight='balanced_subsample',
                            random_state=0)

# Record the start time
start = time.time()
y_train_single_column = np.argmax(Y_train, axis=1)

# Train the classifier
rf.fit(X_train, y_train_single_column)  # Assuming y_train_single_column is your target with single column containing class labels

# Record the end time
end = time.time()

# Print the time taken for training
print(f'Time taken for training: {end - start} seconds')

# Predict the classes and the class probabilities of the test set
y_preds = rf.predict(X_test)
y_probs = rf.predict_proba(X_test)

# Print the predictions and probabilities
print(y_preds)
print(y_probs)


# In[109]:


y_test_single_column = np.argmax(Y_test, axis=1)


# In[50]:


#Dos_sample = X_test.iloc[64].values


# In[ ]:





# In[26]:


#explainer = shap.KernelExplainer(rf_classifier.predict, shap.sample(X_train, 50))
#shap_values = explainer.shap_values(X_test[:5])


# In[27]:


# Get SHAP values for the 34rd sample
#shap_values = explainer.shap_values(X_test.iloc[:50])



# In[37]:


print(y_preds)


# In[28]:


#Dos_sample = X_test.iloc[64].values.reshape(1, -1)


# In[31]:


# Mapping the binary class labels to class names
class_names = ['Normal', 'DoS', 'Probe', 'U2R', 'R2L']

# Predicting the class using RF classifier



#Dos_sample_p1 = X_test.iloc[64]


# In[35]:


#Dos_sample_p1[['dst_host_srv_serror_rate']] =0


# In[36]:


#Dos_sample_p1 = Dos_sample_p1.values.reshape(1, -1)


# In[37]:


#prediction = rf_classifier.predict(Dos_sample_p1)


# In[38]:


#explainer = shap.KernelExplainer(rf_classifier.predict, shap.sample(X_train, 50))


# In[39]:


# Get SHAP values for the 34rd sample
#shap_values = explainer.shap_values(X_test.iloc[:50])



# In[40]:


#print(prediction)


# In[41]:


# Mapping the binary class labels to class names
class_names = ['Normal', 'DoS', 'Probe', 'U2R', 'R2L']
'''
# Predicting the class using RF classifier
prediction = rf_classifier.predict(Dos_sample_p1)
print(prediction)
# Converting binary class labels to class names
predicted_class = class_names[[i for i, x in enumerate(prediction[0]) if x][0]]

print(f"The RF predicted: {predicted_class}")
print("------------------------------------------")

# Getting the actual value from y_test
actual_value = Y_test[34]

# Converting actual binary class labels to class names
actual_class = class_names[[i for i, x in enumerate(actual_value) if x][0]]

# Printing the actual class name
print(f"Actual value: {actual_class}")
'''


# In[43]:


#Dos_sample_p2 = X_test.iloc[64]


# In[44]:


#Dos_sample_p2[['dst_host_srv_serror_rate']] =0


# In[45]:


#Dos_sample_p2[['rerror_rate']] =0


# In[ ]:





# In[46]:


#Dos_sample_p2 = Dos_sample_p2.values.reshape(1, -1)


# In[47]:


# Mapping the binary class labels to class names
class_names = ['Normal', 'DoS', 'Probe', 'U2R', 'R2L']
import lime
from lime.lime_tabular import LimeTabularExplainer 
# Predicting the class using RF classifier
'''
prediction = rf_classifier.predict(Dos_sample_p2)
print(prediction)
# Converting binary class labels to class names
predicted_class = class_names[[i for i, x in enumerate(prediction[0]) if x][0]]

print(f"The RF predicted: {predicted_class}")
print("------------------------------------------")

# Getting the actual value from y_test
actual_value = Y_test[64]

# Converting actual binary class labels to class names
actual_class = class_names[[i for i, x in enumerate(actual_value) if x][0]]

# Printing the actual class name
print(f"Actual value: {actual_class}")
'''


# In[126]:


import warnings
warnings.filterwarnings("ignore")


# In[ ]:

# In[80]:


class_names=['Normal', 'DoS', 'Probe', 'U2R', 'R2L']


x_axis = [0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1]

def waterfall_explanator(sample, instance_index=0):
    # datapoint to explain
    explainer = shap.TreeExplainer(rf)  
    prediction = rf.predict(sample)  # Prediction of the sample
    # Convert the numpy array to a Python list
    prediction_list = prediction[0].tolist()
    # Extract the index accordingly to prediction
    label = [[1,0,0,0,0], [0,1,0,0,0], [0,0,1,0,0], [0,0,0,1,0], [0,0,0,0,1],[0,0,0,0,0]]
    class_index = label.index(prediction_list)
    
    # Generating SHAP values
    shap_values = explainer.shap_values(sample)
    
    # Creating DataFrame of feature importances
    feature_importance = pd.DataFrame({
        'feature': sample.columns,
        'shap_values': shap_values[class_index][instance_index],
    })

    feature_importance['abs_shap_values'] = feature_importance['shap_values'].abs()
    feature_importance.sort_values(by=['abs_shap_values'], ascending=False, inplace=True)
    
    return prediction, feature_importance

def completeness_all(single_class_samples, number_samples, number_of_features_perturbation):
    Bucket = {
    '0.0': 0,
    '0.1':0,
    '0.2':0,
    '0.3':0,
    '0.4':0,
    '0.5':0,
    '0.6':0,
    '0.7':0,
    '0.8':0,
    '0.9':0,
    '1.0':0,

           }
    Counter_all_samples = 0
    counter_samples_changed_class = 0
    print('------------------------------------------------')
    print('Initiating Completeness Experiment')
    print('------------------------------------------------')
    for i in range(number_samples):
        # Select sample
        sample = single_class_samples[i:i+1].copy()
        # Explain the original sample
        u = waterfall_explanator(sample)
        if u is None:  # Skip if the explanation is None (edge case where prediction is [0,0,0,0,0])
            continue
        print(u)
        # Select top k features from the original sample
        top_k_features = u[1]['feature'].head(number_of_features_perturbation).tolist()
        break_condition = False
        for feature in top_k_features:
            for j in range(11):  # 11 steps to include 1.0 (0 to 10)
                perturbation = j / 10.0  # Divide by 10 to get steps of 0.1
                temp_var = sample[feature].copy()
                sample[feature] = perturbation
                v = waterfall_explanator(sample)
                if v is None or not np.array_equal(v[0], u[0]):
                    Bucket[str(perturbation)] += 1  
                    break_condition = True
                    counter_samples_changed_class += 1
                    break
                else:
                    sample[feature] = temp_var  # Restore the original value
            if break_condition:
                break
        Counter_all_samples += 1
        progress = 100 * Counter_all_samples / number_samples
        if progress % 10 == 0:
            print('Progress', progress, '%')
    dict = Bucket
    temp = 0
    for k in dict:
        dict[k] = dict[k] + temp
        temp = dict[k]
    total = number_samples
    y_axis = []
    for k in dict:
        dict[k] = abs(dict[k] - total)
        y_axis.append(dict[k]/total)    
    return(counter_samples_changed_class,Counter_all_samples,y_axis)


import numpy as np
import pandas as pd

# Assume Y_test is a numpy array with shape (n_samples, 5) and X_test is a pandas DataFrame
# If Y_test is not a numpy array, convert it using Y_test = np.array(Y_test)

# Create masks for each class
masks = {
    'Normal': Y_test[:, 0] == 1,
    'DoS': Y_test[:, 1] == 1,
    'Probe': Y_test[:, 2] == 1,
    'U2R': Y_test[:, 3] == 1,
    'R2L': Y_test[:, 4] == 1
}

# Number of samples and features for perturbation to test
number_samples = 1000  # Change accordingly
number_of_features_perturbation = 5  # Change accordingly

# Iterate through each class
for attack_class, mask in masks.items():
    # Extract X_test samples for the current class
    class_samples = X_test[mask]
    
    print(f"Processing {attack_class} samples:")
    
    # Ensure there are enough samples to process
    if class_samples.shape[0] < number_samples:
        print(f"Not enough samples for {attack_class}. Adjusting number of samples to {class_samples.shape[0]}.")
        number_samples = class_samples.shape[0]
    
    # Run the completeness_all function for the current class samples
    changed, total, pp = completeness_all(class_samples, number_samples, number_of_features_perturbation)
    print(pp)
    print(f"Class {attack_class}: {changed} out of {total} samples changed classification.")
    print("----------------------------------------------------")



# In[ ]:



plt.clf()

 
## Update y_axis manually with the results.

# Plot the first line
plt.plot(x_axis,[0.826, 0.818, 0.816, 0.815, 0.814, 0.814, 0.717, 0.716, 0.716, 0.716, 0.716], label='Normal', color='red', linestyle='--', marker='x')

 

 

# Plot the first line
plt.plot(x_axis, y_axis_dos, label='DoS', color='blue', linestyle='--', marker='o')

 

# # Plot the second line
plt.plot(x_axis, y_axis_normal, label='Normal', color='red', linestyle='--', marker='x')

 

# # Plot the third line
plt.plot(x_axis, y_axis_ps, label='Port Scan', color='magenta', linestyle='-.', marker='+')

 

# # Plot the fourth line
# plt.plot(x_axis, y_axis_infiltration, label='Infiltration', color='purple', linestyle='--', marker='p')

 

# # Plot the fifth line
# plt.plot(x_axis, y_axis_bot, label='Bot', color='orange', linestyle='--', marker='h')

 

# # Plot the sixth line
# plt.plot(x_axis, y_axis_web, label='Web Attack', color='magenta', linestyle='--', marker='+')

 

# # Plot the seventh line
# plt.plot(x_axis, y_axis_brute, label='Brute Force', color='cyan', linestyle='--', marker='_')

 

# Enable grid lines (both major and minor grids)
plt.grid()

 

# Customize grid lines (optional)
# plt.grid()

 

# Add labels and a legend
plt.xlabel('Perturbation',fontsize = 18)
plt.ylabel('Samples remaining',fontsize = 18)
plt.legend(fontsize = 12)

 

# Set the title of the plot
# plt.title('Accuracy x Features - SHAP SML')

 

# Show the plot
plt.show()
plt.savefig('GRAPH_PERT_LIME_SML.png')
plt.clf()

 

 

# Enable grid lines (both major and minor grids)
plt.grid()

 

# Customize grid lines (optional)
# plt.grid()

 

# Add labels and a legend
plt.xlabel('Perturbation',fontsize = 18)
plt.ylabel('Samples remaining',fontsize = 18)
plt.legend(fontsize = 12)

 

# Set the title of the plot
# plt.title('Accuracy x Features - SHAP SML')

 

# Show the plot
plt.show()
plt.savefig('GRAPH_PERT_SHAP_CIC.png')
plt.clf()
