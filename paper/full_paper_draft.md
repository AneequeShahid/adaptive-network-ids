# Adaptive Network Intrusion Detection Using Online Learning to Address Concept Drift in CICIDS2017 Traffic Patterns

**Abstract**  
Traditional Network Intrusion Detection Systems (NIDS) rely heavily on signature-based or static machine learning approaches. While highly effective in static environments, these systems often experience severe performance degradation when exposed to novel network attack vectors—a phenomenon known as concept drift. This paper investigates the use of online learning algorithms to maintain high detection accuracy in continuously evolving network traffic. We benchmark static baseline models (Logistic Regression, Random Forest) against online learning models (Hoeffding Tree, Adaptive Random Forest) using a strict chronological temporal split of the CICIDS2017 dataset. This approach accurately simulates real-world concept drift by testing models on attack distributions not seen during training. Furthermore, we explicitly integrate ADWIN and DDM drift detectors to monitor the error rate stream and flag significant data distribution shifts. Our results demonstrate that the Adaptive Random Forest (ARF) achieves an impressive 99.76% accuracy and 99.27% F1 score, outperforming static models and demonstrating rapid recovery following concept drift. This research highlights the necessity of online learning and active drift detection in modern, adaptive cybersecurity frameworks.

## 1. Introduction
With the rapid proliferation of IoT devices and high-speed networks, network traffic volumes and the sophistication of cyberattacks have increased exponentially. Attackers constantly evolve their techniques, resulting in non-stationary network traffic distributions over time. When static Machine Learning (ML) models are deployed in such dynamic environments, their predictive capabilities deteriorate rapidly, limiting their long-term efficacy. To mitigate this "concept drift," Network Intrusion Detection Systems (NIDS) must be capable of continuous, incremental learning. This study proposes and evaluates an adaptive NIDS framework utilizing River ML online learning algorithms, rigorously tested against the CICIDS2017 dataset under simulated chronological drift conditions.

## 2. Related Work
Traditional NIDS heavily rely on signature matching, failing against zero-day exploits. While ML-based NIDS address this by generalizing from traffic features, they are often evaluated using randomized cross-validation on static datasets. This approach inherently leaks future information into the training phase and fails to account for concept drift [1, 2]. Recent literature has explored online learning for cybersecurity, showing that incremental algorithms can update internal decision boundaries without full retraining [3]. However, comprehensive benchmarks comparing online versus static model degradation under strict temporal dataset splits, coupled with explicit drift detection integration (ADWIN/DDM), remain limited.

## 3. Methodology
### 3.1. Dataset Description and Preprocessing
We utilized the CICIDS2017 dataset, containing 2,231,806 rows across five consecutive days, reflecting both benign traffic and varied, modern attacks. Following data cleaning (handling NaNs and infinite values), we employed a `SelectKBest` feature selection strategy to isolate the top 20 most informative features from the original 78. Features were then standardized, and labels were binarily encoded (`BENIGN` vs `ATTACK`). The overall class balance stood at 84.92% benign and 15.08% attack traffic.

### 3.2. Temporal Drift Simulation
To accurately test for concept drift, we implemented a chronological split rather than a random shuffle. Traffic from Monday through Wednesday served as the training set, establishing the base behavior. The test set consisted of Thursday and Friday traffic, deliberately introducing novel attacks (e.g., Web Attacks, Infiltration, DDoS) entirely unseen during the training phase.

### 3.3. Model Selection and Drift Detection
- **Static Baselines:** Logistic Regression and Random Forest (scikit-learn).
- **Online Learning:** Hoeffding Tree and Adaptive Random Forest (River ML).
- **Drift Detectors:** ADWIN and DDM were applied to the continuous error stream to mathematically pinpoint moments of statistical deviation.

## 4. Results and Discussion
The evaluation on the Thursday-Friday temporally shifted test set yielded the following results:

**Table 1: Performance Comparison on Shifted Test Set**

| Model | Type | Accuracy | F1 Score |
| :--- | :--- | :--- | :--- |
| Logistic Regression | Static | 96.51% | 88.79% |
| Random Forest | Static | 98.13% | 94.15% |
| Hoeffding Tree | Online | 99.06% | 97.25% |
| Adaptive Random Forest | Online | 99.76% | 99.27% |

The static baseline models exhibited noticeable degradation when faced with the novel attacks introduced on Thursday and Friday. The static Random Forest achieved a 94.15% F1 score, struggling to classify the new distributions. Conversely, the online learning models adapted successfully. The Adaptive Random Forest (ARF) maintained an exceptional 99.76% accuracy and 99.27% F1 score. Furthermore, the explicit drift detectors successfully flagged the introduction of these novel attacks; ADWIN detected 1 significant concept drift point corresponding with the onset of the new attack patterns.

## 5. Conclusion
This study empirically demonstrates the severe limitations of static NIDS deployments when exposed to concept drift. Online learning models, particularly the Adaptive Random Forest, offer a robust and highly accurate alternative, natively adapting to evolving threat landscapes without requiring expensive offline retraining. Integrating explicit drift detection mechanisms like ADWIN provides valuable interpretability, alerting security analysts to fundamental shifts in network behavior. Future work will explore applying these online paradigms to multi-class intrusion detection and deeper feature streaming selection.

## 6. References
[1] I. Sharafaldin, A. H. Lashkari, and A. A. Ghorbani, "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization," *ICISSP*, 2018.
[2] A. Aldweesh, A. Derhab, and A. Z. Emam, "Deep learning approaches for anomaly-based intrusion detection systems: A survey, taxonomy, and open issues," *Knowledge-Based Systems*, 2020.
[3] V. H. Gomes et al., "Adaptive random forests for evolving data stream classification," *Machine Learning*, 2017.
