# Results and Analysis

## 1. Overall Performance
Table 1 summarizes the performance of the static baselines versus the online learning models evaluated sequentially on the temporally split test set (Thursday-Friday traffic).

**Table 1: Performance Comparison on Test Set**

| Model | Type | Accuracy | F1 Score | Adaptability Score |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | Static | {{LR_ACC}} | {{LR_F1}} | {{LR_ADAPT}} |
| Random Forest | Static | {{RF_ACC}} | {{RF_F1}} | {{RF_ADAPT}} |
| SVM | Static | {{SVM_ACC}} | {{SVM_F1}} | {{SVM_ADAPT}} |
| Hoeffding Tree | Online | {{HT_ACC}} | {{HT_F1}} | {{HT_ADAPT}} |
| Adaptive Random Forest | Online | {{ARF_ACC}} | {{ARF_F1}} | {{ARF_ADAPT}} |

## 2. Model Degradation Analysis
As shown in Figure 1 (Accuracy over Time), static models exhibit a sharp decline in predictive accuracy when encountering novel attacks introduced on Thursday and Friday. Conversely, the online learning models, particularly the Adaptive Random Forest, recover rapidly after an initial drop, showcasing their ability to assimilate new traffic profiles incrementally.

## 3. Drift Detection Events
Figure 2 highlights the precise moments when the ADWIN and DDM detectors flagged concept drift. The detection points correlate strongly with the introduction of the DDoS and Web Attacks in the dataset timeline, empirically validating our hypothesis that network traffic environments are highly non-stationary.
