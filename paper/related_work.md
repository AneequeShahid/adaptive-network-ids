# Related Work

## 1. Traditional IDS Approaches
Traditional Network Intrusion Detection Systems (NIDS), such as Snort and Suricata, heavily rely on signature-based detection. While highly effective against known threats with established signatures, these systems fail to detect zero-day vulnerabilities and novel attack vectors.

## 2. ML-Based IDS
To address the limitations of signature-based systems, researchers have increasingly turned to Machine Learning (ML). Numerous studies have successfully applied algorithms like Random Forests, Support Vector Machines (SVM), and Deep Neural Networks (DNN) to network traffic classification. These systems learn the underlying features of benign and malicious traffic, allowing them to generalize and detect unknown attacks. 

## 3. The Concept Drift Problem in IDS
Despite high accuracy in controlled environments, static ML models face severe degradation in real-world deployments due to "concept drift." Concept drift occurs when the statistical properties of the target variable change over time in unforeseen ways. In cybersecurity, this is driven by attackers continuously evolving their techniques to bypass detection. A static model trained on yesterday's traffic data quickly becomes obsolete against tomorrow's exploits.

## 4. Online Learning for IDS
Online learning (or incremental learning) presents a viable solution to concept drift. Unlike batch learning models that require full retraining on the entire dataset when updated, online learning models process data sequentially (one sample or a small mini-batch at a time) and continuously update their internal state. Recent literature has begun exploring Hoeffding Trees and Adaptive Random Forests for NIDS, showing promising results in maintaining accuracy over streaming data.

## 5. Gaps in Existing Literature
While online learning has been explored, comprehensive benchmarks comparing the degradation rates of static versus online models under strict temporal evaluation on modern datasets (like CICIDS2017) remain limited. This research bridges that gap by explicitly pairing online learning algorithms with explicit drift detection mechanisms (ADWIN, DDM) to quantify adaptability in non-stationary network environments.
