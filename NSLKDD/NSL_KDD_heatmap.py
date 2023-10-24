import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Features from SHAP for different models
shap_features = {
    'ADA_SHAP': ['dst_host_srv_count', 'dst_host_serror_rate', 'src_bytes', 'dst_bytes', 'count', 'dst_host_same_src_port_rate', 'service_http', 'service_private', 'dst_host_diff_srv_rate', 'dst_host_rerror_rate', 'dst_host_count', 'service_telnet', 'duration', 'srv_count', 'dst_host_srv_diff_host_rate', 'serror_rate', 'flag_S0', 'hot', 'service_other', 'service_eco_i'],
    'DNN_SHAP': ['dst_host_serror_rate', 'dst_host_same_srv_rate', 'dst_host_rerror_rate', 'dst_host_srv_count', 'dst_host_same_src_port_rate', 'rerror_rate', 'same_srv_rate', 'logged_in', 'srv_rerror_rate', 'count', 'service_http', 'dst_host_diff_srv_rate', 'dst_host_srv_serror_rate', 'dst_host_count', 'dst_host_srv_rerror_rate', 'diff_srv_rate', 'srv_serror_rate', 'serror_rate', 'service_eco_i'],
    'LGBM_SHAP': ['src_bytes', 'diff_srv_rate', 'dst_host_srv_count', 'same_srv_rate', 'dst_host_same_srv_rate', 'dst_host_serror_rate', 'count', 'dst_bytes', 'dst_host_same_src_port_rate', 'flag_S0', 'flag_RSTOS0', 'service_http', 'dst_host_rerror_rate', 'dst_host_diff_srv_rate', 'logged_in', 'dst_host_count', 'Protocol_type_tcp', 'dst_host_srv_serror_rate', 'srv_count', 'service_private'],
    'SVM_SHAP': ['dst_host_srv_count', 'srv_rerror_rate', 'count', 'service_http', 'service_telnet', 'same_srv_rate', 'dst_host_same_srv_rate', 'srv_serror_rate', 'dst_host_same_src_port_rate', 'dst_host_rerror_rate', 'hot', 'root_shell', 'service_private', 'flag_REJ', 'rerror_rate', 'logged_in', 'dst_host_serror_rate', 'srv_count', 'flag_SF', 'service_ftp_data'],
    'MLP_SHAP': ['dst_host_same_srv_rate', 'dst_host_srv_count', 'dst_host_rerror_rate', 'dst_host_serror_rate', 'dst_host_same_src_port_rate', 'root_shell', 'service_telnet', 'dst_host_count', 'srv_rerror_rate', 'hot', 'same_srv_rate', 'logged_in', 'flag_REJ', 'count', 'srv_count', 'num_file_creations', 'service_http', 'flag_SF', 'dst_host_diff_srv_rate', 'dst_host_srv_serror_rate'],
    'RF_SHAP': ['dst_host_serror_rate', 'dst_bytes', 'src_bytes', 'flag_S0', 'dst_host_same_srv_rate', 'dst_host_srv_count', 'logged_in', 'serror_rate', 'srv_serror_rate', 'dst_host_diff_srv_rate', 'dst_host_rerror_rate', 'flag_SF', 'count', 'dst_host_srv_serror_rate', 'same_srv_rate', 'service_private', 'rerror_rate', 'dst_host_same_src_port_rate', 'diff_srv_rate', 'srv_count'],
    'KNN_SHAP': ['same_srv_rate', 'flag_SF', 'flag_S0', 'dst_host_srv_serror_rate', 'dst_host_serror_rate', 'serror_rate', 'srv_serror_rate', 'dst_host_same_srv_rate', 'dst_host_srv_count', 'logged_in', 'count', 'service_http', 'dst_host_count', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate','service_name','service_klogin','num_shells' ,'service_uucp','label'],
    
 
}

# Features from LIME for different models
lime_features = {
    'ADA_LIME': ['root_shell', 'dst_host_rerror_rate', 'same_srv_rate', 'rerror_rate', 'service_domain_u', 'service_shell', 'service_smtp', 'logged_in', 'dst_host_srv_count', 'service_pm_dump', 'service_telnet', 'dst_host_srv_rerror_rate', 'num_file_creations', 'srv_rerror_rate', 'serror_rate', 'srv_diff_host_rate', 'service_other', 'num_failed_logins', 'service_tftp_u', 'service_private'],
    'DNN_LIME': ['service_aol', 'service_harvest', 'service_tftp_u', 'service_pm_dump', 'service_http', 'service_tim_i', 'serror_rate', 'srv_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_serror_rate', 'service_red_i', 'wrong_fragment', 'hot', 'num_shells', 'count', 'su_attempted', 'root_shell', 'service_rje', 'num_root', 'service_domain'],
    'LGBM_LIME': ['rerror_rate', 'duration', 'dst_host_srv_count', 'same_srv_rate', 'src_bytes', 'wrong_fragment', 'diff_srv_rate', 'service_tftp_u', 'service_tim_i', 'service_shell', 'service_remote_job', 'service_red_i', 'service_finger', 'dst_host_count', 'service_aol', 'dst_host_rerror_rate', 'srv_diff_host_rate', 'service_urh_i', 'dst_host_serror_rate', 'dst_host_diff_srv_rate'],
    'SVM_LIME': ['wrong_fragment', 'hot', 'service_ecr_i', 'root_shell', 'is_guest_login', 'srv_rerror_rate', 'same_srv_rate', 'num_failed_logins', 'num_root', 'srv_count', 'service_sql_net', 'rotocol_type_icmp', 'srv_serror_rate', 'service_kshell', 'service_nntp', 'count', 'service_uucp', 'dst_host_same_srv_rate', 'rerror_rate'],
    'MLP_LIME': ['wrong_fragment', 'service_remote_job', 'srv_rerror_rate', 'service_tim_i', 'root_shell', 'service_ecr_i', 'src_bytes', 'num_shells', 'is_guest_login', 'num_failed_logins', 'urgent', 'count', 'dst_host_count', 'service_other', 'dst_host_same_srv_rate', 'dst_host_serror_rate', 'service_ftp', 'dst_host_rerror_rate', 'num_access_files', 'service_harvest'],
    'RF_LIME': ['dst_host_serror_rate', 'serror_rate', 'srv_serror_rate', 'dst_host_srv_serror_rate', 'service_aol', 'service_harvest', 'service_red_i', 'service_urh_i', 'service_http_', 'su_attempted', 'service_printer', 'service_rje', 'service_ntp_u', 'service_pop_', 'flag_', 'service_sql_net', 'service_nntp', 'num_file_creations', 'service_ssh', 'service_kshell'],
    'KNN_LIME': ['service_vmnet', 'service_bgp', 'service_uucp_path', 'service_courier', 'service_iso_tsap', 'service_whois', 'service_csnet_ns', 'service_nnsp', 'service_daytime', 'service_hostnames', 'service_login', 'service_uucp', 'service_efs', 'service_netbios_dgm', 'service_systat', 'service_exec', 'service_discard', 'wrong_fragment', 'service_name', 'service_klogin'],
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
