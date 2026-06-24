# Methodology

## 1. Dataset Description
The CICIDS2017 dataset represents a realistic network environment with both benign background traffic and up-to-date common attacks. It contains PCAP files and corresponding CSV files with 80+ network traffic features extracted using CICFlowMeter. The dataset spans five consecutive days, where each day incorporates different attack profiles.

## 2. Preprocessing
We handled missing values, infinity values, and duplicate entries to ensure data quality. Infinity values were replaced with NaNs, and subsequently, all rows containing NaNs were removed. Features were standardized using a `StandardScaler` to have zero mean and unit variance, a critical step for distance-based and gradient-based learning algorithms.

## 3. Feature Selection
To mitigate the curse of dimensionality and reduce computational overhead, we employed a feature selection strategy. Using the `SelectKBest` method with `mutual_info_classif`, we identified the top 20 most informative features that share the highest mutual information with the target labels.

## 4. Experimental Setup & Drift Simulation
A core aspect of our methodology is simulating real-world concept drift. Instead of a randomized train-test split, we enforced a strict temporal split. Traffic from Monday to Wednesday was used to train the static baseline models. Traffic from Thursday and Friday, which introduce entirely new attack vectors (e.g., Web Attacks, Infiltration, DDoS), served as the test set. This chronological split accurately tests a model's ability to generalize to unseen, drifting patterns.

## 5. Model Selection
- **Static Baselines:** We selected Random Forest, Logistic Regression, and Support Vector Machines (SVM) due to their prevalence in traditional NIDS literature.
- **Online Learning:** We utilized the Hoeffding Tree and Adaptive Random Forest (ARF) from the `river` library. These models process data incrementally, allowing them to adapt to evolving data distributions.

## 6. Drift Detection
We integrated explicit drift detection mechanisms, specifically ADWIN (Adaptive Windowing) and DDM (Drift Detection Method). These detectors monitor the error rate stream of the models, triggering alerts when the error rate shifts significantly, signaling a potential concept drift.
