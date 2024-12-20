

### E-XAI: Evaluating Black-Box Explainable AI Frameworks for Network Intrusion Detection

### Abstract
The escalating frequency of intrusions on networked systems has spurred a wave of innovative research in artificial intelligence (AI) techniques tailored for intrusion detection systems (IDS). The burgeoning need for comprehensible and interpretable AI models is evident among security analysts tasked with overseeing these IDS to ensure the integrity of their networks. Addressing this need, we introduce an all-encompassing framework designed to rigorously evaluate black-box XAI methods applied to network IDS.

Our study encapsulates an examination of both global and local scopes of these enigmatic XAI methods in the realm of network intrusion detection. We meticulously analyze six diverse evaluation metrics, specifically tailored for two renowned black-box XAI techniques, SHAP and LIME. These evaluation metrics are artfully woven, encapsulating essential elements from both network security and AI paradigms.

The robustness and versatility of our XAI evaluation framework is validated through extensive trials involving three prominent network intrusion datasets and seven distinct AI methods, each exuding unique attributes. We wholeheartedly extend an invitation to the network security community, offering unrestricted access to our codes, aspiring to establish a foundational XAI framework for network IDS.

Our empirical findings unveil the intrinsic limitations and unparalleled strengths inherent in contemporary black-box XAI methods when meticulously applied to network IDS, offering invaluable insights and serving as a catalyst for future exploratory endeavors in this domain.


### Framework

![image](https://github.com/ogarreche/XAI_metrics/assets/55901425/ea663c33-657b-42ee-a865-99fac4a2535d)

In a high-level fashion, our framework starts by loading
well renowned network intrusion datasets that contain
traffic logs from the network of interest. After collecting
network’s data, our framework feeds such data into different
black-box AI models where these models detect intrusions
and normal traffic via constructing a multi-class classification
problem. In this classification problem, each class represents
one possible network intrusion (e.g., denial of service,
port-scanning, and brute force attacks). After predicting
traffic instances using the black-box AI model, the following
step is to pass these predictions into XAI methods to
generate the explanations of the instances and the associated
predictions. Then, our framework evaluate these XAI
methods via leveraging the six XAI evaluation metrics (i.e.,
descriptive accuracy, sparsity, efficiency, stability, robustness,
and completeness).

### Key Features
- Comprehensive evaluation of black-box XAI methods for network IDS.
- Detailed analysis using six distinct metrics, offering insights from network security and AI perspectives.
- Extensive evaluations performed on three widely recognized network intrusion datasets.
- Open-source codes made accessible to bolster collaborative efforts and advancements in the field.


### Evaluation Results
0. Prelude - Feature Selection
We initialized our evaluation by assigning importance scores to intrusion features, which were then used in subsequent experiments. Each AI model generated global explanations via SHAP and LIME. During trials for descriptive accuracy, sparsity, Stability, Robustness and Completeness. we experimented with the removal of top-k features, determining their impact on the model’s performance. Stability was evaluated through repeated trials. Robustness and completeness were assessed on a local level, analyzing individual explanations during integrity attacks and perturbations. The efficiency experiment assessed computational time required for SHAP and LIME under varied scenarios.

1. Descriptive Accuracy
We commenced with the evaluation of the descriptive accuracy of XAI methods on network intrusion datasets. Figures 2-3 exhibit the descriptive accuracy for SHAP and LIME under three distinct datasets. The insights gained underscore SHAP’s superiority over LIME in terms of global explainability.



![DA Graphs](DA_Graphs.png)



![DA Table](DA_Table.png)



Main Insights:
A noticeable drop in accuracy with the removal of features was observed predominantly in NSL-KDD dataset.
SHAP outperformed LIME, as detailed in Table 7.
An anomaly was detected where accuracy remained consistent or increased under top feature removal, raising speculations regarding equal feature contributions or the ‘‘curse of dimensionality’’.


2. Sparsity
The subsequent metric, sparsity, assessed the distribution of feature importance scores. Figures 4-5 illustrate the sparsity for SHAP and LIME. SHAP again surfaced as the victor, evidencing superior performance in terms of concentrated explanations.



![Sparsity Table](Sparsity_Table.png)




![Sparsity Graphs](Sparsity_Graphs.png)

Main Insights:
SHAP's concentration of explanations in fewer intrusion features underscores its low sparsity and elevated explainability.
The AUC curves’ exponential growth shape, especially evident in SHAP’s performance, underscores the concentration of explanation in top features.
LIME, however, exhibited a linear relationship in AUC curves in some cases, indicating a spread of explanation across more features (Table 8).
Summary of Tables and Figures
Table 6 showcases AI models that exhibited a significant accuracy drop, segmented by dataset and XAI method.
Table 7 outlines the quantitative results of descriptive accuracy, providing insights into the nuanced performance of AI models under the scrutinization of XAI methods across varied datasets.
Table 8 provides a quantitative overview of the sparsity metric, illuminating the area under the curve for distinct AI models, XAI methods, and datasets.
Figures 2-3 illustrate the descriptive accuracy of SHAP and LIME XAI methods under the three evaluated network intrusion datasets.
Figures 4-5 depict the sparsity results for both XAI methods, offering a comparative lens to evaluate the concentration of XAI explanations with respect to intrusion features.

3. Robustness:

The robustness of an XAI method can be
defined as the ability of that XAI method in generating
the same explanations under small perturbation in the
intrusion features. Such perturbation can be due to
computational errors or an adversarial attack. For this
work, we adapted the adversarial model of the work [24].
This adversarial model consists of training an extremely
biased model that will rely only on one feature to base
its prediction on, and training another model with all
the features plus a new feature that is engineered to fool
the XAI explanation method. Based on such two trained
models, XAI will generate an explanation of a normal
sample that is convincing to a security analyst while in
reality the real sample was hidden from the XAI method.
This can compromise the integrity of the framework
via disconnecting the explanation from the underlying
behavior, and thus disguise an attack as normal network
traffic.

An example of DoS sample from NSL-KDD dataset for robustness experiment using SHAP. In (a), the feature list
(with flow duration as the top feature) under biased explanation is shown. In (b), the list (with engineered feature as the top
feature) after adversarial model’s classification is shown

![image](https://github.com/ogarreche/XAI_metrics/assets/55901425/31ec7d63-7f8c-47da-9f01-76e9c64017a5)


FIGURE 8: The percentage of data samples in CICIDS-2017 dataset for which biased and unrelated features appear in top-3
features (according to LIME and SHAP rankings of feature importance) for the biased classifier (in (a) and (c)) and adversarial
classifier (in (b) and (d)) that uses one uncorrelated feature.


![image](https://github.com/ogarreche/XAI_metrics/assets/55901425/25e18abe-c7fa-43bc-9c1c-38cf0102ba50)

FIGURE 9: The percentage of data samples in RoEduNet-SIMARGL2021 dataset for which biased and unrelated features
appear in top-3 features (according to LIME and SHAP rankings of feature importance) for the biased classifier (in (a) and (c))
and adversarial classifier (in (b) and (d)) that uses one uncorrelated feature.

![image](https://github.com/ogarreche/XAI_metrics/assets/55901425/9227c722-7cb4-4d08-bfd0-4be1d5a8daea)

FIGURE 10: The feature occurence percentage of data samples in NSL-KDD dataset for which biased and unrelated features
appear in top-3 features (according to LIME and SHAP rankings of feature importance) for the biased classifier (in (a) and (c))
and adversarial classifier (in (b) and (d)).

![image](https://github.com/ogarreche/XAI_metrics/assets/55901425/6e20b1c7-35c7-4e3f-a83d-47d308c5b3ee)

FIGURE 11: Robustness sensitivity for LIME for all three network intrusion datasets.

![image](https://github.com/ogarreche/XAI_metrics/assets/55901425/7410fd57-d032-4841-8595-4d893c5e904b)


FIGURE 12: Robustness sensitivity for SHAP for all three network intrusion datasets.

![image](https://github.com/ogarreche/XAI_metrics/assets/55901425/1e5cf443-dd3f-4615-bb28-5286a13ec86e)



Main Insights:   

-LIME performs better than SHAP for the NSL-KDD
dataset, bringing more exposure to the biased feature. On
the other hand, LIME performs worse than SHAP for
CICIDS-2017 and RoEduNet-SIMARGL2021 datasets with
fewer occurrences of the biased feature. Overall, SHAP
technique performed better than LIME for two of the three
datasets selected for this study in this experiment.  See Figures 8, 9, and 10.


-Overall, although LIME and SHAP are both
vulnerable to such attacks starting at a certain threshold, our
experiments show that LIME is slightly more robust since
its initial threshold towards attacks of this kind being higher
compared to SHAP. See Figures 11 and 12.

4. Efficiency

The efficiency of an XAI method can be
defined as how much time the XAI method takes
to generate an explanation. This metric is important
because it measures the applicability of the XAI method
in real-world systems since it is preferable that the
explanations are generated quickly rather than slowly
for practical purposes. As the ultimate goal is to aid
security analysts, the expectation is to be able to
generate accurate XAI explanations in real-time to detect
intrusions in a timely manner.

TABLE 11: The efficiency (amount of time in hours) for generating SHAP and LIME explanations for different AI models and
different number of samples.

![image](https://github.com/ogarreche/XAI_metrics/assets/55901425/9eb345df-d07b-4860-88c0-13b0210ae9b4)


Main Insights: 

-Given these results, we have the following recommendations
from computational efficiency perspective: (i) the best choice
for SHAP is to pair it with LGBM, RF, or DNN for global
explanations, (ii) the usage of SHAP is better than LIME
for local explanations, and (iii) adapting LIME for global
explanations is more efficient compared to SHAP. See Table 11.

5. Completeness

The completeness characteristic of an
XAI method refers to the ability of XAI method
in providing a correct explanation for every possible
network traffic sample, including corner cases. If the
XAI method is not complete, this opens the door for
the intruders to exploit and trick the XAI method to
create degenerated results. We emphasize that if an XAI
is complete, it automatically becomes more robust (i.e.,
its reliability for the end user is increased because it can
detect whether the explanation is valid). Nonetheless,
while the measure for completeness is by checking that
every sample has a valid explanation, the robustness
metric in this work refers to how resistant an XAI
framework is when facing an adversarial attack.




![image](https://github.com/ogarreche/XAI_metrics/assets/55901425/0b05cffe-2b01-43ca-a7f4-9d3f888f3ac0)

FIGURE 13: An example of completeness experiment on DoS sample from RoEduNet-SIMARGL2021 dataset. This experiment
is using SHAP method. In (a), we show the original XAI explanation. In (b), we show it after the top feature is perturbed. In
(c), we show it after the second top feature is perturbed.

![image](https://github.com/ogarreche/XAI_metrics/assets/55901425/f80f5505-6bd1-490f-8a6f-559604374e0b)

FIGURE 14: The LIME analysis of completeness experiment
on the DoS sample from RoEduNet-SIMARGL2021 dataset.
In first quadrant, we show the original explanation. In second
quadrant, we show it after the top feature is perturbed. In third
quadrant, we show it after the second top feature is perturbed.



![image](https://github.com/ogarreche/XAI_metrics/assets/55901425/fd90af3f-8a5d-4bb7-bfc1-74cf5a454402)

FIGURE 16: Percentage of remaining network traffic samples after removing the samples in which the intrusion class changed
under different perturbation levels in intrusion features. We observe that higher perturbations tend to change classes for both
LIME and SHAP for all three datasets.

TABLE 12:  The percentage of samples that are complete (i.e.,
with valid explanations) for each intrusion class using SHAP
and LIME for RoEduNet-SIMARGL2021 dataset.

![image](https://github.com/ogarreche/XAI_metrics/assets/55901425/d3c5fd3b-6afa-4c03-9ce4-d5d3221400e4)

TABLE 13: The percentage of samples that are complete
(i.e., with valid explanations) for each intrusion class using
CICIDS-2017 dataset.

![image](https://github.com/ogarreche/XAI_metrics/assets/55901425/0a7afa64-1841-44c4-932f-d5f5df1ec8e7)

TABLE 14: The percentage of samples that are complete for
each class using NSL-KDD dataset.

![image](https://github.com/ogarreche/XAI_metrics/assets/55901425/d203678e-ddb5-477a-9cfe-310ae4331944)





Main Insights:

-Since we use the same method for
feature perturbation for SHAP and LIME, we conclude that
LIME tends to under-perform in terms of global completeness
when dealing with multiclass classification with more
intrusion labels. For NSL-KDD dataset, Table 14 show that
the performance of SHAP and LIME differs significantly
for this dataset. In every class, SHAP performs noticeably
better than LIME, especially in the ‘‘Normal" and ‘‘DoS"
categories. This discrepancy confirms SHAP’s effectiveness
for complex and diverse datasets like NSL-KDD. To sum
up, our experiments highlight that SHAP is more complete
than LIME, indicating that it can provide more thorough
and dependable interpretability in intricate classification
scenarios.

6. Stability:

The stability metric measures how consistent
the XAI method is in generating its explanations. This is
measured via checking how many features are common
among different running trials with the same conditions.
An XAI method with higher stability can be trusted more
by the security analyst in network intrusion detection
process.

TABLE 9: The global stability evaluation metric of XAI
methods. We measure such stability by calculating the
percentage of features that intersect in different runs among
the total number of features for several network traffic
instances.

![image](https://github.com/ogarreche/XAI_metrics/assets/55901425/1fa73a88-eaec-4c4a-bc08-3a5a38c92e91)

TABLE 10: The local stability evaluation metric of XAI
methods. We measure such stability by calculating the
percentage of features that intersect in different runs among
the total number of features for single network traffic
instance.


Main Insights:

As expected, local stability is higher
than global due to the fact that it runs the same instance
instead of a collection of samples, which are prone to have
more differences by pure chance. From our experimental
results (see Tables 9-10), we notice that LIME and
SHAP perform roughly the same with respect to this
category. For CICIDS-2017 dataset, global LIME has better
performance compared to SHAP for four models (RF,
DNN, LGBM, and MLP) while SHAP performed better
for only two models (SVM and KNN). For CICIDS-2017,
local LIME performed better in two models (RF and
LGBM) and it was tied with SHAP for the other five
models. For RoEduNet-SIMARGL2021 dataset, global
LIME outperforms SHAP one time (LGBM) while the same
is true for SHAP (DNN) and the remaining results are tied.
For RoEduNet-SIMARGL2021, local SHAP outperforms
LIME one time (DNN) and the remaining results are tied.
For NSL-KDD dataset, there is a noticeable difference in the
stability of LIME and SHAP explanations across different
AI models. In terms of global stability, LIME showed better
stability with RF, DNN, and MLP, while SHAP was more stable with LGBM, ADA, SVM, and KNN. However, the
local stability of LIME and SHAP varied with different
models, indicating a nuanced performance that depends
heavily on the specific combination of the dataset, AI model,
and XAI explanation method.

7. Common Features across XAI Methods: We provide a
heatmap that shows the common top features across different
AI models and XAI methods for all of the three datasets.
Figure 17 shows such heatmaps for all three datasets. It
shows that there is very little overlap between the top features found by SHAP and LIME. Perturbing these features, we
found that the top features of SHAP cause a discernible
class transition, highlighting their importance. On the other
hand, perturbing LIME’s top features does not result in a
noteworthy variation, suggesting that LIME may be less
significant in the classification context. This difference is
further supported by the global completeness in Tables 13-14,
which shows that higher shift in resulting intrusion class in
response to perturbations in SHAP’s features than in LIME’s,
indicating that SHAP is more reliable at identifying pertinent
intrusion features.



![image](https://github.com/ogarreche/XAI_metrics/assets/55901425/5b71e08c-4962-4f83-8f10-7fe648c4f4c6)

FIGURE 17: A heatmap analysis that shows the common top features across different AI models and the two XAI methods
(SHAP and LIME) for all of the three studied datasets. The number of matched features are shown for each pair.

8. CNN experiment:

To successfully apply CNN, we
lean into the work of [Zhang et al., 2019]. This work transforms the network intrusion data into
the array data format to apply it to the CNN model. Such a model consists of one Convolution 1D
layer, pooling 1D layer, fully connected layer, and ReLU activation function. In sequence, the model
is trained and evaluated for all datasets. See Table 11 below for performances of CNN for all of
our three datasets. Such a table shows that CNN achieves high performance for CICIDS-2017 and
RoEduNet-SIMARGL2021, while it has lower performance for NSL-KDD dataset. Note that Table
11 shows all performance metrics.

![image](https://github.com/ogarreche/XAI_metrics/blob/main/CNN_DA.png)

Figure 18: Descriptive Accuracy using the CNN model
for the CICIDS-2017 dataset. Note that the plot indicates
evidence that the most important features might not be
influential because there is no significant drop in accuracy
for both XAI techniques.


![image](https://github.com/ogarreche/XAI_metrics/blob/main/CNN_SPAR.png)

FIGURE 19: Sparsity metric using the CNN model for the
CICIDS-2017 dataset. Note that the SHAP XAI method
outperforms LIME because it has more area under the curve,
and its curve ascends sooner.

Main Insights:

Based on the results, we can conclude that LIME is
better suited to work with the CNN setup used for this experiment. The reasons for this verdict are
mainly the memory efficiency problem found when using SHAP paired with CNN, which makes the
combination almost unusable and impractical. In the other spectrum, LIME does not display the
same issue. Considering the XAI metrics, both LIME and SHAP perform the same in Descriptive
Accuracy in which there is no significant drop in accuracy for both cases, which also might indicate
the case that CNN might not be the best option for explainability when compared to the other seven
models analyzed (e.g., see the response to the Reviewer 2). In the case of Sparsity, SHAP performed
better than LIME, indicating it selected better features. Analyzing the Time Efficiency, LIME is
roughly 30 times faster than SHAP in the same conditions analyzed. Unfortunately, we could not
derive the Stability experiment for SHAP due to its memory complexity issue. Nonetheless, LIME
performed 54.54% for Stability.

9. Statistical Analysis:



Table 10: This table shows the pairwise statistical test results between every pair of AI models by Wilcoxon signed rank test.
Statistically better method (p < 0.05) shown in bold (both marked bold if there is significance and the median of the accuracies
are the same. Only one is marked bold if there is significance and one model has a higher median). On the left, the CICIDS-2017
testbed is shown. In the middle, the RoEduNet-SIMARGL2021 testbed is shown. On the right, the NSL-KDD testbed is shown.
For all testbeds, our method is statistically sound.

![image](https://github.com/ogarreche/XAI_metrics/blob/main/wilcoxon.png)

Main Insights:

Considering the three datasets, we observe
that most pair models reject the null hypothesis (p < 0.05).
Therefore, there is evidence that one model performs better
than the other in most cases. Thus, we can separate the tests
into two groups: The group with statistical significance (i.e.,
p < 0.05). And the group that does not have statistical
significance (p > 0.05). We can further divide the first
group into two subgroups by analyzing the medians of their
array of accuracy scores of the estimator for each run of the
cross-validation method. The first subgroup, indeed, has a
model that is performing better (i.e., one model has a higher
median than the other one). For this case, we highlighted
the best model in bold in Table 17. However, for the second
subgroup (i.e., both models display the same median), we
cannot know which is better, but only that they are statistically
different enough due to the evidence shown by the Wilcoxon
test. In this case, we highlighted both models in bold in
Table 10.

### Datasets:

Download one of the datasets. 

RoEduNet-SIMARGL2021: https://www.kaggle.com/datasets/7f91274fa3074d53e983f6eb7a7b24ad1dca136ca967ad0ebe48955e246c24ee 

CICIDS-2017: [https://www.kaggle.com/datasets/cicdataset/cicids2017](https://www.kaggle.com/datasets/usmanshuaibumusa/cicids-17)

NSL-KDD: [https://www.unb.ca/cic/datasets/nsl.html](https://www.kaggle.com/datasets/hassan06/nslkdd)

### How to use the programs:

Inside the CICIDS or SIMARGL or NSLKDD folder you will find programs for each model used in this paper. Each one of these programs outputs:
***(Note: For the CNN programs refer to CNN folder)***

  - The accuracy for the AI model.
  - The values for y_axis for the sparsity (that will need to be copied and pasted into the general_sparsity_graph.py).
  - Top features in importance order (that will be needed to rerun these same programs to obtain new Accuracy values for Descriptive Accuracy. Take note of the values as you use less features and input these values in  general_desc_acc_graph.py ).
  - For Stability, run the programs 3x or more and input the obtained top k features in general_stability_comparison.py).
    

Descriptive Accuracy:

  - To generate Descriptive Accuracy Graphs, see the code general_desc_acc_graph.py

Sparsity:

  - To generate Sparsity Graphs, see the code general_sparsity_graph.py

Stability:

  - To generate the Stability metrics, see the code general_stability_comparison.py

Robustness:
  - Inside the Robustness folder, firts run the code: threshold_CIC.py to generate a csv file. Then run analyze_threshold_CIC_LIME.py to generate the Robustness Sensitivity graph. and run the program RF_SHAP_CIC_bar.ipynb to generate the robustness bar graphs. 

Completeness:

  - Inside the CICIDS or SIMARGL or NSLKDD folder run the code RF_LIME_COM_SML_CHART.ipynb or RF_SHAP_COM_SML_CHART.ipynb as an example. 

Efficiency:

  - Inside the CICIDS or SIMARGL or NSLKDD folder you will find programs for each model used in this paper. They output the time spent to generate the SHAP or LIME evaluation for  k samples. We can just set up in the program the k value and take note of the time spent.

Heatmap: 

  - To generate heatmaps, see the code [dataset]_heatmap.py located in each dataset folder.

### Citation:

Please cite our work if it was useful to you. 

https://ieeexplore.ieee.org/abstract/document/10433134

### Example:

Inside the SIMARGL folder there is an Example.ipynb with an example to generate Descriptive Accuracy, Sparsity, Stability and Efficieny. For Robustness and Completeness see its instructions above!



### Note

1) The programs were tested on linux. If using windows, you might run in the error: "UnicodeDecodeError: 'utf-8' codec can't decode byte 0x96 in position 22398: invalid start byte
", please refer to: https://github.com/ogarreche/Ensemble_Learning_2_Levels_IDS/issues/1 


2) FOR THE CICIDS dataset: I suggest going to the Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv file and manually changing the following labels with the weird characters (you can do control+F and change all). The idea here is just to group the different labels into similar groups:
Web Attack � Sql Injection to Web Attack
Web Attack � Brute Force  to Web Attack
Web Attack � XSS to Web Attack
And do the same that have the  �. 

3) Why am seeing sligthly different results for robustness? 
It could be due to different versions of the python packages, sometimes when a package is updated the way that it handles number can vary slightly causing slightly different results. In my opinion, this is not necessarily a bad thing since the results are still consistent.

4) Some of these programs take a long time to complete (over a few days).


  
### References:

Wilcoxon: 

  - To generate the statistical analysis, refer to the folder "wilcoxon" and run the programs inside. To obtain the p-values run the .ipynb programs, and for the medians run the .py programs.

Robustness based on: https://github.com/dylan-slack/Fooling-LIME-SHAP/

CNN experiments based on: [https://github.com/dylan-slack/Fooling-LIME-SHAP/](https://github.com/mahbubhimel/Intrusion-Detection-System-Using-Convolutional-Neural-Network)

Six metrics evaluated are based on: https://arxiv.org/abs/1906.02108

[1] A. L. Buczak and E. Guven, ‘‘A survey of data mining and machine learning
methods for cyber security intrusion detection,’’ IEEE Communications
surveys & tutorials, vol. 18, no. 2, pp. 1153–1176, 2015.
[2] A. S. Dina and D. Manivannan, ‘‘Intrusion detection based on machine
learning techniques in computer networks,’’ Internet of Things, vol. 16, p.
100462, 2021.
[3] J. Kim, N. Shin, S. Y. Jo, and S. H. Kim, ‘‘Method of intrusion detection
using deep neural network,’’ in 2017 IEEE international conference on big
data and smart computing (BigComp). IEEE, 2017, pp. 313–316.
[4] C. Tang, N. Luktarhan, and Y. Zhao, ‘‘Saae-dnn: Deep learning method on
intrusion detection,’’ Symmetry, vol. 12, no. 10, p. 1695, 2020.
[5] P. Tao, Z. Sun, and Z. Sun, ‘‘An improved intrusion detection algorithm
based on ga and svm,’’ Ieee Access, vol. 6, pp. 13 624–13 631, 2018.
[6] H. Deng, Q.-A. Zeng, and D. P. Agrawal, ‘‘Svm-based intrusion detection
system for wireless ad hoc networks,’’ in 2003 IEEE 58th Vehicular
Technology Conference. VTC 2003-Fall (IEEE Cat. No. 03CH37484),
vol. 3. IEEE, 2003, pp. 2147–2151.
[7] M. A. Ferrag, L. Maglaras, A. Ahmim, M. Derdour, and H. Janicke,
‘‘Rdtids: Rules and decision tree-based intrusion detection system for
internet-of-things networks,’’ Future internet, vol. 12, no. 3, p. 44, 2020.
[8] M. Al-Omari, M. Rawashdeh, F. Qutaishat, M. Alshira’H, and N. Ababneh,
‘‘An intelligent tree-based intrusion detection model for cyber security,’’
20 VOLUME 11, 2023
O. Arreche et al.: Evaluating Black-Box Explainable AI Frameworks for Network Intrusion Detection
Journal of Network and Systems Management, vol. 29, no. 2, pp. 1–18,
2021.
[9] N. B. Amor, S. Benferhat, and Z. Elouedi, ‘‘Naive bayes vs decision trees in
intrusion detection systems,’’ in Proceedings of the 2004 ACM symposium
on Applied computing, 2004, pp. 420–424.
[10] R. Panigrahi, S. Borah, M. Pramanik, A. K. Bhoi, P. Barsocchi, S. R. Nayak,
and W. Alnumay, ‘‘Intrusion detection in cyber–physical environment
using hybrid naïve bayes—decision table and multi-objective evolutionary
feature selection,’’ Computer Communications, vol. 188, pp. 133–144,
2022.
[11] A. K. Balyan, S. Ahuja, U. K. Lilhore, S. K. Sharma, P. Manoharan,
A. D. Algarni, H. Elmannai, and K. Raahemifar, ‘‘A hybrid intrusion
detection model using ega-pso and improved random forest method,’’
Sensors, vol. 22, no. 16, p. 5986, 2022.
[12] S. Waskle, L. Parashar, and U. Singh, ‘‘Intrusion detection system using
pca with random forest approach,’’ in 2020 International Conference on
Electronics and Sustainable Communication Systems (ICESC). IEEE,
2020, pp. 803–808.
[13] S. Arisdakessian, O. A. Wahab, A. Mourad, H. Otrok, and M. Guizani,
‘‘A survey on iot intrusion detection: Federated learning, game theory,
social psychology and explainable ai as future directions,’’ IEEE Internet
of Things Journal, 2022.
[14] S. I. Sabev, ‘‘Integrated approach to cyber defence: Human in the loop.
technical evaluation report,’’ Information & Security: An International
Journal, vol. 44, pp. 76–92, 2020.
[15] A. Das and P. Rad, ‘‘Opportunities and challenges in explainable artificial
intelligence (xai): A survey,’’ arXiv preprint arXiv:2006.11371, 2020.
[16] B. Mahbooba, M. Timilsina, R. Sahal, and M. Serrano, ‘‘Explainable
artificial intelligence (xai) to enhance trust management in intrusion
detection systems using decision tree model,’’ Complexity, vol. 2021, 2021.
[17] S. Patil, V. Varadarajan, S. M. Mazhar, A. Sahibzada, N. Ahmed, O. Sinha,
S. Kumar, K. Shaw, and K. Kotecha, ‘‘Explainable artificial intelligence
for intrusion detection system,’’ Electronics, vol. 11, no. 19, p. 3079, 2022.
[18] S. R. Islam, W. Eberle, S. K. Ghafoor, A. Siraj, and M. Rogers, ‘‘Domain
knowledge aided explainable artificial intelligence for intrusion detection
and response,’’ arXiv preprint arXiv:1911.09853, 2019.
[19] E. Roponena, J. Kampars, J. Grabis, and A. Gail ̄ıtis, ‘‘Towards a
human-in-the-loop intelligent intrusion detection system,’’ in CEUR
Workshop Proceedings, 2022, pp. 71–81.
[20] D. Han, Z. Wang, W. Chen, K. Wang, R. Yu, S. Wang, H. Zhang, Z. Wang,
M. Jin, J. Yang et al., ‘‘Anomaly detection in the open world: Normality
shift detection, explanation, and adaptation,’’ in 30th Annual Network and
Distributed System Security Symposium (NDSS), 2023.
[21] L. Dhanabal and S. Shantharajah, ‘‘A study on nsl-kdd dataset for intrusion
detection system based on classification algorithms,’’ International journal
of advanced research in computer and communication engineering, vol. 4,
no. 6, pp. 446–452, 2015.
[22] J. Dieber and S. Kirrane, ‘‘Why model why? assessing the strengths and
limitations of lime,’’ arXiv preprint arXiv:2012.00093, 2020.
[23] A. Warnecke, D. Arp, C. Wressnegger, and K. Rieck, ‘‘Evaluating
explanation methods for deep learning in security,’’ in 2020 IEEE european
symposium on security and privacy (EuroS&P). IEEE, 2020, pp. 158–174.
[24] D. Slack, S. Hilgard, E. Jia, S. Singh, and H. Lakkaraju, ‘‘Fooling lime
and shap: Adversarial attacks on post hoc explanation methods,’’ in
Proceedings of the AAAI/ACM Conference on AI, Ethics, and Society, 2020,
pp. 180–186.
[25] M. T. Ribeiro, S. Singh, and C. Guestrin, ‘‘"why should I trust you?":
Explaining the predictions of any classifier,’’ CoRR, vol. abs/1602.04938,
2016. [Online]. Available: http://arxiv.org/abs/1602.04938
[26] S. M. Lundberg, G. Erion, H. Chen, A. DeGrave, J. M. Prutkin, B. Nair,
R. Katz, J. Himmelfarb, N. Bansal, and S.-I. Lee, ‘‘From local explanations
to global understanding with explainable ai for trees,’’ Nature Machine
Intelligence, vol. 2, no. 1, pp. 2522–5839, 2020.
[27] M. Wang, K. Zheng, Y. Yang, and X. Wang, ‘‘An explainable machine
learning framework for intrusion detection systems,’’ IEEE Access, vol. 8,
pp. 73 127–73 141, 2020.
[28] C. Wu, A. Qian, X. Dong, and Y. Zhang, ‘‘Feature-oriented design of
visual analytics system for interpretable deep learning based intrusion
detection,’’ in 2020 International Symposium on Theoretical Aspects of
Software Engineering (TASE), 2020, pp. 73–80.
[29] M.-E. Mihailescu, D. Mihai, M. Carabas, M. Komisarek, M. Pawlicki,
W. Hołubowicz, and R. Kozik, ‘‘The proposition and evaluation of
the roedunet-simargl2021 network intrusion detection dataset,’’ Sensors,
vol. 21, no. 13, p. 4319, 2021.
[30] R. Panigrahi and S. Borah, ‘‘A detailed analysis of cicids2017 dataset
for designing intrusion detection systems,’’ International Journal of
Engineering & Technology, vol. 7, no. 3.24, pp. 479–482, 2018.
[31] D. Han, Z. Wang, W. Chen, Y. Zhong, S. Wang, H. Zhang, J. Yang, X. Shi,
and X. Yin, ‘‘Deepaid: Interpreting and improving deep learning-based
anomaly detection in security applications,’’ 09 2021.
[32] Y. Mirsky, T. Doitshman, Y. Elovici, and A. Shabtai, ‘‘Kitsune: An
ensemble of autoencoders for online network intrusion detection,’’ 2018.
[33] S. Neupane, J. Ables, W. Anderson, S. Mittal, S. Rahimi, I. Banicescu, and
M. Seale, ‘‘Explainable intrusion detection systems (x-ids): A survey of
current methods, challenges, and opportunities,’’ 2022.
[34] B. E. Strom, A. Applebaum, D. P. Miller, K. C. Nickels, A. G. Pennington,
and C. B. Thomas, ‘‘Mitre att&ck: Design and philosophy,’’ in Technical
report. The MITRE Corporation, 2018.
[35] C. B. Lee, C. Roedel, and E. Silenok, ‘‘Detection and characterization
of port scan attacks,’’ Univeristy of California, Department of Computer
Science and Engineering, 2003.
[36] Kurniabudi, D. Stiawan, Darmawijoyo, M. Y. Bin Idris, A. M. Bamhdi, and
R. Budiarto, ‘‘Cicids-2017 dataset feature analysis with information gain
for anomaly detection,’’ IEEE Access, vol. 8, pp. 132 911–132 921, 2020.
[37] M.-E. Mihailescu, D. Mihai, M. Carabas, M. Komisarek, M. Pawlicki,
W. Hołubowicz, and R. Kozik, ‘‘The proposition and evaluation of
the roedunet-simargl2021 network intrusion detection dataset,’’ Sensors,
vol. 21, no. 13, 2021.
[38] B. Stone-Gross, M. Cova, L. Cavallaro, B. Gilbert, M. Szydlowski,
R. Kemmerer, C. Kruegel, and G. Vigna, ‘‘Your botnet is my botnet:
analysis of a botnet takeover,’’ in Proceedings of the 16th ACM conference
on Computer and communications security, 2009, pp. 635–647.
[39] Y. Chen, Q. Lin, W. Wei, J. Ji, K.-C. Wong, and C. A. Coello Coello,
‘‘Intrusion detection using multi-objective evolutionary convolutional
neural network for internet of things in fog computing,’’ Knowledge-Based
Systems, vol. 244, p. 108505, 2022. [Online]. Available: https://www.
sciencedirect.com/science/article/pii/S0950705122002179
[40] V. Gorodetski and I. Kotenko, ‘‘Attacks against computer network:
Formal grammar-based framework and simulation tool,’’ in International
Workshop on Recent Advances in Intrusion Detection. Springer, 2002,
pp. 219–238.
[41] W. Lee, S. Stolfo, and K. Mok, ‘‘A data mining framework for building
intrusion detection models,’’ in Proceedings of the 1999 IEEE Symposium
on Security and Privacy (Cat. No.99CB36344), 1999, pp. 120–132.
[42] T.-S. Chou, K. Yen, and J. Luo, ‘‘Network intrusion detection design
using feature selection of soft computing paradigms,’’ Computational
Intelligence - CI, vol. 47, 01 2008.
[43] A. Khan, H. Kim, and B. Lee, ‘‘M2mon: Building an mmio-based security
reference monitor for unmanned vehicles.’’ 2021.
[44] S. R. Hussain, I. Karim, A. A. Ishtiaq, O. Chowdhury, and E. Bertino,
‘‘Noncompliance as deviant behavior: An automated black-box
noncompliance checker for 4g lte cellular devices,’’ in Proceedings
of the 2021 ACM SIGSAC Conference on Computer and Communications
Security, 2021, pp. 1082–1099.
[45] O. Mirzaei, R. Vasilenko, E. Kirda, L. Lu, and A. Kharraz, ‘‘Scrutinizer:
Detecting code reuse in malware via decompilation and machine learning,’’
in Detection of Intrusions and Malware, and Vulnerability Assessment:
18th International Conference, DIMVA 2021, Virtual Event, July 14–16,
2021, Proceedings 18. Springer, 2021, pp. 130–150.
[46] S. Lukacs, D. H. Lutas, A. V. COLESA et al., ‘‘Strongly isolated malware
scanning using secure virtual containers,’’ Aug. 25 2015, uS Patent
9,117,081.
[47] A. Kim, M. Park, and D. H. Lee, ‘‘Ai-ids: Application of deep learning to
real-time web intrusion detection,’’ IEEE Access, vol. 8, pp. 70 245–70 261,
2020.
[48] M. Botacin, F. Ceschin, R. Sun, D. Oliveira, and A. Grégio, ‘‘Challenges
and pitfalls in malware research,’’ Computers & Security, vol. 106, p.
102287, 2021.
[49] I. Amit, J. Matherly, W. Hewlett, Z. Xu, Y. Meshi, and Y. Weinberger,
‘‘Machine learning in cyber-security - problems, challenges and data
sets,’’ 2018. [Online]. Available: https://arxiv.org/abs/1812.07858
[50] N. Capuano, G. Fenza, V. Loia, and C. Stanzione, ‘‘Explainable artificial
intelligence in cybersecurity: A survey,’’ IEEE Access, vol. 10, pp.
93 575–93 600, 2022.
[51] B. M. Greenwell, ‘‘pdp: an r package for constructing partial dependence
plots.’’ R J., vol. 9, no. 1, p. 421, 2017.
[52] A. Goldstein, A. Kapelner, J. Bleich, and E. Pitkin, ‘‘Peeking inside
the black box: Visualizing statistical learning with plots of individual
VOLUME 11, 2023 21
O. Arreche et al.: Evaluating Black-Box Explainable AI Frameworks for Network Intrusion Detection
conditional expectation,’’ journal of Computational and Graphical
Statistics, vol. 24, no. 1, pp. 44–65, 2015.
[53] E. Anderssen, K. Dyrstad, F. Westad, and H. Martens, ‘‘Reducing
over-optimism in variable selection by cross-model validation,’’
Chemometrics and intelligent laboratory systems, vol. 84, no. 1-2,
pp. 69–74, 2006.
[54] ‘‘Flow information elements - nprobe 10.1 documentation.’’ [Online].
Available: https://www.ntop.org/guides/nprobe/flow_information_
elements.html
[55] Ahlashkari, ‘‘Cicflowmeter/readme.txt at
master · ahlashkari/cicflowmeter,
https://github.com/ahlashkari/cicflowmeter/blob/master/readme.txt,’’ Jun
2021. [Online]. Available: https://github.com/ahlashkari/CICFlowMeter/
blob/master/ReadMe.txt
[56] B. Claise, ‘‘Cisco systems netflow services export version 9,’’ Tech. Rep.,
2004.
[57] I. Sharafaldin, A. Gharib, A. H. Lashkari, and A. A. Ghorbani, ‘‘Towards a
reliable intrusion detection benchmark dataset,’’ Software Networking, vol.
2018, no. 1, pp. 177–200, 2018.
[58] M. Tavallaee, E. Bagheri, W. Lu, and A. A. Ghorbani, ‘‘A detailed analysis
of the kdd cup 99 data set,’’ in 2009 IEEE Symposium on Computational
Intelligence for Security and Defense Applications, 2009, pp. 1–6.
[59] R. ZHAO, ‘‘Nsl-kdd,’’ 2022. [Online]. Available: https://dx.doi.org/10.
21227/8rpg-qt98
[60] C. A. Stewart, V. Welch, B. Plale, G. C. Fox, M. Pierce, and T. Sterling,
‘‘Indiana university pervasive technology institute,’’ techreport, 2017.
[61] S. M. Lundberg and S.-I. Lee, ‘‘A unified approach to interpreting model
predictions,’’ Advances in neural information processing systems, vol. 30,
2017.
[62] A. Yulianto, P. Sukarno, and N. A. Suwastika, ‘‘Improving adaboost-based
intrusion detection system (ids) performance on cic ids 2017 dataset,’’ in
Journal of Physics: Conference Series, vol. 1192, no. 1. IOP Publishing,
2019, p. 012018.
[63] W. Li, P. Yi, Y. Wu, L. Pan, and J. Li, ‘‘A new intrusion detection system
based on knn classification algorithm in wireless sensor network,’’ Journal
of Electrical and Computer Engineering, vol. 2014, 2014.
[64] J. O. Mebawondu, O. D. Alowolodu, J. O. Mebawondu, and A. O.
Adetunmbi, ‘‘Network intrusion detection system using supervised
learning paradigm,’’ Scientific African, vol. 9, p. e00497, 2020.
[65] D. Jin, Y. Lu, J. Qin, Z. Cheng, and Z. Mao, ‘‘Swiftids: Real-time intrusion
detection system based on lightgbm and parallel intrusion detection
mechanism,’’ Computers & Security, vol. 97, p. 101984, 2020.
[66] I. Insights, ‘‘42 Cyber Attack Statistics by Year: A
Look at the Last Decade,’’ https://sectigostore.com/blog/
42-cyber-attack-statistics-by-year-a-look-at-the-last-decade/, February
2020, [Online; accessed 10-March-2023].
[67] N. Moustafa and J. Slay, ‘‘Unsw-nb15: a comprehensive data set for
network intrusion detection systems (unsw-nb15 network data set),’’
in 2015 military communications and information systems conference
(MilCIS). IEEE, 2015, pp. 1–6.
[68] U. T. Repository, ‘‘UMass Trace Repository,’’ http://traces.cs.
umass.edu/index.php/Network/Network, 2021, [Online; accessed on
21-November-2022].
[69] Y. Dong, W. Guo, Y. Chen, and X. Xing, ‘‘Towards the detection of
inconsistencies in public security vulnerability reports.’’
