"""
* Methods to create graphs for f1 accuracy on perturbation task graphs.
"""
import pandas as pd
import numpy as np

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score

from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv('data/threshold_results_CIC_shap.csv', index_col=0)
f1s, fsts, scnds, thrds = [], [], [], []

for trial in np.unique(df['trial']):
	relevant_runs = df[df.trial == trial]

	yhat = relevant_runs['yhat']
	y = relevant_runs['y']

	# need to flip classes (we interpret 0 as ood in code but refer to it as 1 in paper)
	yhat = 1 - yhat
	y = 1 - y

	pct_first = relevant_runs['pct_occur_first'].values[0]
	pct_second = relevant_runs['pct_occur_second'].values[0]
	pct_third = relevant_runs['pct_occur_third'].values[0]

	f1 = f1_score(y, yhat)

	f1s.append(f1)
	fsts.append(pct_first)
	scnds.append(pct_second)
	thrds.append(pct_third)

ax = plt.axes()
plt.ylim(-.05,1.05)
plt.xlim(0,1)

plt.xlabel("F1 score on OOD task", fontsize=18)
plt.ylabel("Ocurrence of biased feature as 1st", fontsize=18)

sns.scatterplot(f1s, fsts, ax=ax)
plt.savefig("0A_ROBUSTNESS_shap_f1_first.png")


# Define the number of bins and the range of F1 scores
num_bins = 10  # Adjust as needed
f1_min, f1_max = min(f1s), max(f1s)
bin_width = (f1_max - f1_min) / num_bins


# Create bins and calculate the mean for each bin
bin_means = []
for i in range(num_bins):
    bin_start = f1_min + i * bin_width
    bin_end = bin_start + bin_width
    mask = (f1s >= bin_start) & (f1s < bin_end)
    bin_mean = np.mean(np.array(fsts)[mask])
    bin_means.append(bin_mean)

# Plot the trend line as the mean of binned points
bin_centers = [f1_min + i * bin_width + bin_width / 2 for i in range(num_bins)]
plt.plot(bin_centers, bin_means, "r--")

# Save the plot
plt.savefig("0A_ROBUSTNESS_SHAP_f1_first_with_binned_trend.png")
