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

# importing library for plotting
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




# In[18]:

#Uncomment For top 20 features from SHAP Analysis

# Specify your selected features. Note that you'll need to modify this list according to your final processed dataframe
#selected_features = ["dst_host_serror_rate","serror_rate","flag_S0","dst_host_rerror_rate","flag_SF","dst_host_srv_count"
                   #,"dst_host_same_srv_rate","count","dst_host_same_src_port_rate","dst_host_diff_srv_rate","diff_srv_rate",
                  #   "rerror_rate","srv_serror_rate","src_bytes","srv_count","dst_host_srv_serror_rate","Protocol_type_tcp","logged_in","Protocol_type_udp","wrong_fragment","label"]
                     

# Select those features from your dataframe
#newdf = newdf[selected_features]
#newdf_test = newdf_test[selected_features]

# Now your dataframe only contains your selected features.


# In[19]:


# creating a dataframe with multi-class labels (Dos,Probe,R2L,U2R,normal)
multi_data = newdf.copy()
multi_label = pd.DataFrame(multi_data.label)

multi_data_test=newdf_test.copy()
multi_label_test = pd.DataFrame(multi_data_test.label)


# In[20]:


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


# In[21]:


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


# In[22]:


multi_data.drop(labels= [ 'label'], axis=1, inplace=True)
multi_data
multi_data_test.drop(labels= [ 'label'], axis=1, inplace=True)
multi_data_test


# In[23]:


y_train_multi= multi_data[['intrusion']]
X_train_multi= multi_data.drop(labels=['intrusion'], axis=1)

print('X_train has shape:',X_train_multi.shape,'\ny_train has shape:',y_train_multi.shape)

y_test_multi= multi_data_test[['intrusion']]
X_test_multi= multi_data_test.drop(labels=['intrusion'], axis=1)

print('X_test has shape:',X_test_multi.shape,'\ny_test has shape:',y_test_multi.shape)


# In[24]:


from collections import Counter

label_counts = Counter(y_train_multi['intrusion'])
print(label_counts)


# In[25]:


from sklearn.preprocessing import LabelBinarizer

y_train_multi = LabelBinarizer().fit_transform(y_train_multi)
y_train_multi

y_test_multi = LabelBinarizer().fit_transform(y_test_multi)
y_test_multi


# In[26]:


Y_train=y_train_multi.copy()
X_train=X_train_multi.copy()

Y_test=y_test_multi.copy()
X_test=X_test_multi.copy()


# In[27]:


print(Y_test)


# In[28]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import time
# Create RandomForestClassifier instance with modified parameters
rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=10, min_samples_leaf=5, class_weight='balanced_subsample', random_state=0)

# Assume 'X_train' is your training data and 'X_test' your test data

# Get feature names from the training set
feature_names = X_train.columns.tolist()

# Reorder the test set to match the training set
X_test = X_test[feature_names]

# Wrap RandomForestClassifier with MultiOutputClassifier
multi_target_rf = MultiOutputClassifier(rf)
start=time.time()
# Training the model
multi_target_rf.fit(X_train.values, Y_train)
end=time.time()
time_taken = end - start
print(f'Time taken for training: {time_taken} seconds')
start=time.time()
# Now you can predict the test set results
preds = multi_target_rf.predict(X_test)
end=time.time()
time_taken = end - start
print(f'Time taken for pred: {time_taken} seconds')



from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier


# In[75]:


print('Defining the model')
print('--------------------------------------------------')
rf = RandomForestClassifier(max_depth = 5,  n_estimators = 10, min_samples_split = 2, n_jobs = -1)
#------------------------------------------------------------------------------
model_rf = rf.fit(X_train.values,Y_train)



preds = rf.predict(X_test)


pred_labels = np.argmax(preds, axis=1)



# Convert Y_test back to its original format
y_test = np.argmax(Y_test, axis=1)


'''

correctly_classified_indices = np.where(pred_labels == y_test)[0]
misclassified_indices = np.where(pred_labels != y_test)[0]


misclassified_indices


# In[33]:

'''
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import numpy as np

preds=np.argmax(preds, axis=1)
y_true_multiclass = np.argmax(Y_test, axis=1)
confusion = confusion_matrix(y_true_multiclass, preds)

# Binarize the output for AUC
lb = LabelBinarizer()
lb.fit(y_true_multiclass)
y_test_bin = lb.transform(y_true_multiclass)
y_pred_bin = lb.transform(preds)

# Iterate through each class and calculate the metrics
class_names = ['Normal','DoS', 'Probe', 'R2L', 'U2R']
for i in range(len(class_names)):
    TP = confusion[i, i]
    FP = confusion[:, i].sum() - TP
    FN = confusion[i, :].sum() - TP
    TN = confusion.sum() - TP - FP - FN
    
    # Call your metrics functions
    Acc = ACC(TP, TN, FP, FN)
    Precision = PRECISION(TP, FP)
    Recall = RECALL(TP, FN)
    F1_score = F1(Recall, Precision)
    Balanced_accuracy = BACC(TP, TN, FP, FN)
    Matthews = MCC(TP, TN, FP, FN)
    
    # AUC_ROC calculation
    AUC_ROC = roc_auc_score(y_test_bin[:, i], y_pred_bin[:, i])
    
    # Print metrics
    print(f'Metrics for: {class_names[i]}')
    print('Accuracy: ', Acc)
    print('Precision: ', Precision)
    print('Recall: ', Recall)
    print('F1: ', F1_score)
    print('BACC: ', Balanced_accuracy)
    print('MCC: ', Matthews)
    print('AUC_ROC: ', AUC_ROC)
    print()

# AUC_ROC total
print('AUC_ROC total: ', roc_auc_score(y_test_bin, y_pred_bin, multi_class='ovr'))
print('---------------------------------------------------------------------------------')



Y_train = Y_train.flatten()





# In[35]:
'''

import lime
import lime.lime_tabular


# In[36]:


# Create a Lime explainer object
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    training_labels=Y_train,
    feature_names=X_train.columns.tolist(),
    class_names=class_names, 
    mode='classification'
)


# In[99]:


correct_instance_df = X_test.iloc[[correctly_classified_indices[1]]]
misclassified_instance_df = X_test.iloc[[misclassified_indices[0]]]

misclassified_exp = explainer.explain_instance(
    misclassified_instance_df.values[0], 
    rf.predict, 
    num_features=7, 
    top_labels=1
)


# In[100]:


#misclassified_exp.show_in_notebook(show_table=True)
print(y_test[misclassified_indices[0]], pred_labels[misclassified_indices[0]])

print(misclassified_exp.available_labels())


misclassified_exp.as_list()

# In[101]:


misclassified_exp.as_list(label=4)


# In[ ]:


correct_exp.show_in_notebook(show_table=True)


# In[46]:


exp = explainer.explain_instance(misclassified_instance_df.values[0], multi_target_rf.predict_proba, num_features=5)
 
exp.show_in_notebook(show_table=True)




# In[144]:


# Use KernelExplainer for model agnostic
explainer = shap.KernelExplainer(model_rf.predict, shap.sample(X_train, 500))

# Calculate Shap values on a small sample of test data
small_X_test = X_test[:500]
shap_values = explainer.shap_values(small_X_test)



shap_values = explainer(small_X_test)

# In[153]:


predictions = model_rf.predict(X_test)
print(predictions)


# In[154]:


predict_probabilities = model_rf.predict_proba(X_test)
print(predict_probabilities)


# In[155]:


explainer = shap.TreeExplainer(model_rf)
shap_values = explainer.shap_values(X_test[:5])

for i, s in enumerate(shap_values):
    print(f"Shape for class {i}: {np.array(s).shape}")


# In[156]:


instance_index = 2  # this represents the third instance in your test dataset
class_index = 0     # for demonstration, we're using class 2

shap.force_plot(
    explainer.expected_value[class_index], 
    shap_values[class_index][instance_index], 
    X_test.iloc[instance_index]
)


# In[157]:


instance_index = 2  # for demonstration, we're taking the third instance in your test dataset
class_index = 0   # for demonstration, we're using class 2

# Extracting explanation for the specific instance and class
explanation = shap.Explanation(
    values=shap_values[class_index][instance_index],
    base_values=explainer.expected_value[class_index],
    data=X_test.iloc[instance_index],
    feature_names=X_test.columns.tolist()
)

# Plotting the waterfall plot
shap.waterfall_plot(explanation)


# In[158]:


class_names = ['Normal','DoS', 'Probe', 'R2L', 'U2R']


# In[136]:


from lime.lime_tabular import LimeTabularExplainer
explainer = LimeTabularExplainer(X_train.values,
                                 feature_names=X_train.columns.tolist(),
                                 class_names=list(range(5)), # Assuming num_classes is the number of unique classes
                                 mode='classification')
instance = X_test.iloc[2].values
exp = explainer.explain_instance(instance, 
                                 model_rf.predict_proba, 
                                 class_names=class_names, 
                                 num_features=X_train.shape[1])
exp.show_in_notebook(show_table=True, show_all=False)


# In[97]:


print(model_rf.predict_proba(instance.reshape(1, -1)))


# In[163]:


# Adjusting for the class of interest
class_index = 0  # for 'Probe'
binary_class_names = [class_names[class_index], f'NOT {class_names[class_index]}']

# Create the LIME explainer
explainer = LimeTabularExplainer(
    X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=['Normal','DoS', 'Probe', 'R2L', 'U2R'],
    mode='classification'
)

def predict_proba_wrapper(data):
    # Get the predictions using the model
    probas = model_rf.predict_proba(data)
    
    # Convert the list of arrays into a 2D numpy array
    stacked_probas = np.column_stack([p[:, 1] for p in probas])
    
    return stacked_probas
instance = X_test.iloc[2].values
exp = explainer.explain_instance(instance, 
                                predict_proba_wrapper, 
                                top_labels=3,  # We're interested in the top 1 class (our specified class)
                                num_features=5)  # Restrict to 5 features
exp.show_in_notebook(show_table=True, show_all=False)





# In[33]:


import matplotlib.pyplot as plt
import matplotlib.pylab as pl

# Create a new matplotlib Figure and Axes
fig, ax = pl.subplots(1,1)

# Plot SHAP summary on the created Axes
shap.summary_plot(shap_values, small_X_test, feature_names=multi_data.columns, class_names=['Normal', 'DoS', 'Probe', 'R2L', 'U2R'])
# Save the figure
plt.savefig('shap_summary_plot_RF.png', bbox_inches='tight')
plt.close(fig)

'''

'''
import lime
import lime.lime_tabular

# Create a Lime explainer object
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    training_labels=Y_train,
    feature_names=X_train.columns.tolist(),
    class_names=class_names, 
    mode='classification'
)
def get_explanation(instance):
    # Reshape the instance to have a batch size of 1
    instance_reshaped = instance.reshape(1, -1)
    
    # Get the explanation using the predict method
    exp = explainer.explain_instance(instance, lambda x: multi_target_rf.predict(x.reshape(-1, len(instance))), num_features=len(X_train.columns))
    return dict(exp.as_list())

# Assuming misclassified_instance_df is a DataFrame containing the instances you want to explain
explanations = [get_explanation(instance) for instance in X_test[:2000].values]

# Then you can aggregate the individual explanations to get a global view.
# For instance, to get the average importance of each feature:
feature_importances = {}
for explanation in explanations:
    for feature, importance in explanation.items():
        if feature not in feature_importances:
            feature_importances[feature] = []
        feature_importances[feature].append(importance)

# Calculate the mean importance for each feature
global_importances = {feature: np.mean(importances) for feature, importances in feature_importances.items()}

# Print the global importances
for feature, importance in global_importances.items():
    print(f'{feature}: {importance}')


import shap
import numpy as np

# Use KernelExplainer for model agnostic
explainer = shap.KernelExplainer(multi_target_rf.predict, shap.sample(X_train, 10))

# Calculate Shap values on a small sample of test data
small_X_test = X_test[:50]
shap_values = explainer.shap_values(small_X_test)


# Assuming shap_values is a list of arrays, we iterate over them to get the mean absolute SHAP value for each output
mean_shap_values_per_output = [np.mean(np.abs(shap_values[i]), axis=0) for i in range(len(shap_values))]

# Get the feature names 
feature_names = X_train.columns

# Iterate over each output and get the feature importance
for i in range(len(mean_shap_values_per_output)):
    feature_importance = dict(zip(feature_names, mean_shap_values_per_output[i]))

    # Sort the features by importance
    sorted_feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

    # Print the sorted feature importances
    print(f"Output {i}")
    for feature, importance in sorted_feature_importance:
        print(f"{feature}: {importance}")
'''

'''

import numpy as np
import lime
import lime.lime_tabular



# ... (include necessary imports for your model and data)
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    training_labels=Y_train,
    feature_names=X_train.columns.tolist(),
    class_names=class_names, 
    mode='classification'
)
def get_explanation(instance):
    exp = explainer.explain_instance(instance, lambda x: multi_target_rf.predict(x), num_features=len(X_train.columns))
    return dict(exp.as_list())

# Generate explanations for a subset of the test set
explanations = [get_explanation(instance) for instance in X_test[:500].values]

# Aggregate individual explanations to compute global feature importance values
feature_importances = {}
for explanation in explanations:
    for feature, importance in explanation.items():
        if feature not in feature_importances:
            feature_importances[feature] = []
        feature_importances[feature].append(importance)

epsilon = 1e-10

# Normalize the importance values to the range [-1, 1]
normalized_importances = {feature: [(2*(x - min(importances))/(max(importances) - min(importances) + epsilon) - 1) for x in importances] for feature, importances in feature_importances.items()}

# Loop to calculate the MAZ score for varying interval sizes
interval_sizes = [0.00, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
average_maz_scores = []

for r in interval_sizes:
    histograms = {feature: np.histogram(importances, bins=np.linspace(-1, 1, num=100)) for feature, importances in normalized_importances.items()}
    
    maz_scores = {}
    for feature, histogram in histograms.items():
        maz = sum(histogram[0][(histogram[1][:-1] >= -r) & (histogram[1][1:] <= r)]) / sum(histogram[0])
        maz_scores[feature] = maz

    # Calculate and store the average MAZ score for this interval size
    average_maz = np.mean(list(maz_scores.values()))
    average_maz_scores.append(average_maz)

# Returning the array of average MAZ scores
print(average_maz_scores)

import shap
import numpy as np

# ... (include necessary imports for your model and data)

# Initialize the KernelExplainer
explainer = shap.KernelExplainer(multi_target_rf.predict, shap.sample(X_train, 10))

# Calculate Shap values on a small sample of test data
small_X_test = X_test[:50]
shap_values = explainer.shap_values(small_X_test)

# Aggregate individual explanations to compute global feature importance values
feature_importances = {}
for i in range(len(X_train.columns)):
    feature_importances[X_train.columns[i]] = [sv[i] for sv in shap_values[0]]

epsilon = 1e-10

# Normalize the importance values to the range [-1, 1]
normalized_importances = {feature: [(2*(x - min(importances))/(max(importances) - min(importances) + epsilon) - 1) for x in importances] for feature, importances in feature_importances.items()}

# Loop to calculate the MAZ score for varying interval sizes
interval_sizes = [0.00, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]
average_maz_scores = []

for r in interval_sizes:
    histograms = {feature: np.histogram(importances, bins=np.linspace(-1, 1, num=100)) for feature, importances in normalized_importances.items()}
    
    maz_scores = {}
    for feature, histogram in histograms.items():
        maz = sum(histogram[0][(histogram[1][:-1] >= -r) & (histogram[1][1:] <= r)]) / sum(histogram[0])
        maz_scores[feature] = maz

    # Calculate and store the average MAZ score for this interval size
    average_maz = np.mean(list(maz_scores.values()))
    average_maz_scores.append(average_maz)

# Print the average MAZ scores for each interval size
for r, score in zip(interval_sizes, average_maz_scores):
    print(f"Interval size: {r}, Average MAZ score: {score}")


import shap
import numpy as np

def get_top_features(model, X_train_sample, X_test_sample, feature_names):
    explainer = shap.KernelExplainer(model.predict, X_train_sample)
    shap_values = explainer.shap_values(X_test_sample)
    
    # Sum the absolute SHAP values across all outputs for each feature
    aggregated_shap_values = np.sum([np.abs(shap_values[i]) for i in range(len(shap_values))], axis=0)
    
    # Get the mean absolute SHAP value across all outputs for each feature
    mean_aggregated_shap_values = np.mean(aggregated_shap_values, axis=0)
    
    # Create a dictionary of feature names and their mean absolute SHAP values
    feature_importance = dict(zip(feature_names, mean_aggregated_shap_values))
    
    # Sort the features by importance
    sorted_feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Return top 20 features
    return [feature[0] for feature in sorted_feature_importance[:20]]

# Define small_X_test for your experiment
small_X_test = X_test[:100]

# Run the function 3 times and store the top features in separate arrays
array1 = get_top_features(multi_target_rf, shap.sample(X_train, 100), small_X_test, X_train.columns)
array2 = get_top_features(multi_target_rf, shap.sample(X_train, 100), small_X_test, X_train.columns)
array3 = get_top_features(multi_target_rf, shap.sample(X_train, 100), small_X_test, X_train.columns)

# Compute the intersection of top features across the 3 runs
common_features = set(array1).intersection(set(array2)).intersection(set(array3))

# Calculate the fraction of the intersection relative to top 20
fraction_common = len(common_features) / 20.0
print(f"Fraction of common features in top 20 across 3 runs: {fraction_common:.3f}")
'''

import lime
import lime.lime_tabular
import numpy as np
import re

# Extracts only the feature name without threshold
def extract_feature_name(feature_string):
    pattern = re.compile(r"([a-z_]+)")
    match = pattern.match(feature_string)
    if match:
        return match.group(1)
    return None

def get_top_lime_features(model, X_train_sample, X_test_sample, feature_names):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train_sample,
        training_labels=Y_train,
        feature_names=feature_names,
        class_names=class_names, 
        mode='classification'
    )
    
    feature_importances = {}
    
    for instance in X_test_sample.values:
        exp = explainer.explain_instance(instance, lambda x: model.predict(x), num_features=len(feature_names))
        explanation = dict(exp.as_list())

        for feature, importance in explanation.items():
            feature_name = extract_feature_name(feature)  # Only extract the feature name
            if feature_name:
                if feature_name not in feature_importances:
                    feature_importances[feature_name] = []
                feature_importances[feature_name].append(np.abs(importance))

    global_importances = {feature: np.mean(importances) for feature, importances in feature_importances.items()}
    sorted_features = sorted(global_importances.items(), key=lambda x: -x[1])

    return [feature[0] for feature in sorted_features[:20]]

small_X_test = X_test[:500]
array1 = get_top_lime_features(multi_target_rf, X_train.values, small_X_test, X_train.columns)
print(array1)
array2 = get_top_lime_features(multi_target_rf, X_train.values, small_X_test, X_train.columns)
print(array2)
array3 = get_top_lime_features(multi_target_rf, X_train.values, small_X_test, X_train.columns)

common_features = set(array1).intersection(set(array2)).intersection(set(array3))
fraction_common = len(common_features) / 20.0
print(f"Fraction of common features in top 20 across 3 runs: {fraction_common:.3f}")
