import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Features from SHAP for different models
shap_features = {
    'ADA_SHAP': ['IN_BYTES', 'TCP_WIN_SCALE_IN', 'PROTOCOL', 'TCP_WIN_MAX_IN', 'TCP_WIN_MSS_IN'],
    'DNN_SHAP': ['TCP_WIN_MAX_IN', 'TCP_WIN_MIN_IN', 'TCP_WIN_SCALE_IN', 'L4_SRC_PORT', 'PROTOCOL'],
    'LGBM_SHAP': ['LAST_SWITCHED', 'L4_DST_PORT', 'FIRST_SWITCHED', 'L4_SRC_PORT', 'FLOW_ID'],
    'SVM_SHAP': ['TCP_WIN_MSS_IN', 'TCP_WIN_MAX_IN', 'TCP_WIN_SCALE_IN', 'TCP_WIN_MIN_IN', 'L4_DST_PORT'],
    'MLP_SHAP': ['TCP_WIN_MIN_IN', 'TCP_WIN_SCALE_IN', 'TCP_WIN_MAX_IN', 'L4_DST_PORT', 'L4_SRC_PORT'],
    'RF_SHAP': ['TCP_WIN_MAX_IN', 'TCP_WIN_SCALE_IN', 'TCP_WIN_MSS_IN', 'TCP_WIN_MIN_IN', 'OUT_PKTS'],
    'KNN_SHAP': ['TCP_WIN_MAX_IN', 'TCP_WIN_SCALE_IN', 'TCP_WIN_MIN_IN', 'TCP_WIN_MSS_IN', 'L4_DST_PORT'],
    
 
}

# Features from LIME for different models
lime_features = {
    'ADA_LIME': ['TCP_WIN_SCALE_IN', 'FIRST_SWITCHED', 'TCP_WIN_MAX_OUT', 'TCP_WIN_MAX_IN', 'TCP_WIN_SCALE_OUT'],
    'DNN_LIME': ['TCP_WIN_MAX_OUT', 'TCP_WIN_MIN_IN', 'TCP_WIN_MIN_OUT', 'TCP_WIN_SCALE_IN', 'TCP_WIN_SCALE_OUT'],
    'LGBM_LIME': ['FLOW_DURATION_MILLISECONDS', 'TCP_WIN_MAX_OUT', 'DST_TOS', 'MAX_IP_PKT_LEN', 'LAST_SWITCHED'],
    'SVM_LIME': ['TCP_WIN_MIN_IN', 'TCP_WIN_SCALE_IN', 'TCP_WIN_MSS_IN', 'TCP_WIN_SCALE_OUT', 'TCP_FLAGS'],
    'MLP_LIME': ['TCP_WIN_MIN_IN', 'TCP_WIN_SCALE_IN', 'TCP_WIN_MAX_IN', 'L4_DST_PORT', 'L4_SRC_PORT'],
    'RF_LIME': ['DST_TOS', 'FLOW_DURATION_MILLISECONDS', 'TCP_WIN_SCALE_IN', 'FLOW_ID', 'TCP_WIN_SCALE_OUT'],
    'KNN_LIME': ['TCP_WIN_MIN_IN', 'TCP_WIN_MAX_OUT', 'DST_TOS', 'SRC_TOS', 'TCP_WIN_MAX_IN'],
}

# Combine all model names
all_models = list(shap_features.keys()) + list(lime_features.keys())

# Compute the number of common features between each pair of models
common_features_matrix = pd.DataFrame(
    [[len((set(shap_features.get(row, [])) & set(lime_features.get(col, []))) |
        (set(shap_features.get(col, [])) & set(lime_features.get(row, []))) |
        (set(shap_features.get(row, [])) & set(shap_features.get(col, []))) |
        (set(lime_features.get(row, [])) & set(lime_features.get(col, []))))
    for col in all_models] for row in all_models],
    index=all_models,
    columns=all_models
)

# Plot a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(common_features_matrix, annot=True, cmap="Blues", fmt="d", cbar=True)
plt.title('Number of Common Features Between SHAP and LIME by Model')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.show()
