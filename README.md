#  ML Sampling Techniques vs Classifier Performance

##  Project Overview

This project studies how different **sampling strategies** affect the
performance of machine-learning classifiers on an **imbalanced
dataset**.

Imbalanced data is common in real-world problems (e.g., fraud detection,
medical diagnosis), and improper handling can lead to biased models. I:

-   Split data correctly into train and test sets
-   Balance only the training data
-   Apply five sampling strategies
-   Train five classifiers
-   Compare performance using accuracy and visualization

------------------------------------------------------------------------

##  Dataset

-   **Name:** Credit Card Dataset\
-   **Target Variable:** `Class`\
-   Highly imbalanced binary classification problem.

------------------------------------------------------------------------

##  Methodology

To avoid evaluation bias and data leakage, the following pipeline was
strictly followed:

1.  **Train/Test Split First**

    -   80% training, 20% testing
    -   Stratified by class labels.

2.  **Balancing**

    -   Only the training data was balanced using **RandomOverSampler**.
    -   The test set was left untouched.

3.  **Sampling Techniques**

    -   Simple Random Sampling
    -   Stratified Sampling
    -   Systematic Sampling
    -   Bootstrap Sampling
    -   Cluster Sampling

4.  **Models Evaluated**

    -   Random Forest
    -   K-Nearest Neighbors (KNN)
    -   Naive Bayes
    -   Multi-Layer Perceptron (MLP)
    -   Support Vector Machine (SVM)

5.  **Evaluation Metric**

    -   Accuracy on the held-out test set.

------------------------------------------------------------------------

## Sampling Techniques

| Technique      | Description                     |
|----------------|---------------------------------|
| Simple Random  | Random subset of records        |
| Stratified     | Preserves class proportions     |
| Systematic     | Every k-th sample               |
| Bootstrap      | Sampling with replacement       |
| Cluster        | Random chunk selection          |

------------------------------------------------------------------------

## Machine-Learning Models

| Model         | Purpose                    |
|-------------- |----------------------------|
| Random Forest | Ensemble decision trees    |
| KNN           | Distance-based classifier  |
| Naive Bayes   | Probabilistic model        |
| MLP           | Neural network             |
| SVM           | Margin-based classifier    |


------------------------------------------------------------------------

##  Results

###  Accuracy (%) Table

## Accuracy Results

| Model ↓ / Sampling → | Sampling1_Simple | Sampling2_Stratified | Sampling3_Systematic | Sampling4_Bootstrap | Sampling5_Cluster |
|---------------------|------------------|---------------------|---------------------|--------------------|------------------|
| **M1_RandomForest** | 99.35 | 99.35 | 99.35 | 99.35 | 99.35 |
| **M2_KNN**          | 96.77 | 96.77 | 96.77 | 96.77 | 96.77 |
| **M3_NaiveBayes**   | 96.13 | 94.19 | 95.48 | 94.19 | 95.48 |
| **M4_MLP**          | 97.42 | 97.42 | 93.55 | 97.42 | 97.42 |
| **M5_SVM**          | 72.90 | 69.03 | 69.03 | 50.97 | 69.03 |

------------------------------------------------------------------------

##  Visualizations

-   Heatmap
-   <img width="867" height="672" alt="image" src="https://github.com/user-attachments/assets/e5018c13-d1c6-4017-a5f2-0c6187a3ae3a" />

-   Line plot
-   <img width="984" height="590" alt="image" src="https://github.com/user-attachments/assets/3961f9fe-ae3c-4539-adc6-bcf746015358" />

-   Box plot
-   <img width="790" height="590" alt="image" src="https://github.com/user-attachments/assets/63d81e88-ab2a-4709-a3f9-c250038c89ad" />

------------------------------------------------------------------------

##  Discussion

-   Random Forest consistently achieved the highest accuracy (\~99%)
    across all sampling strategies.
-   KNN and MLP also performed strongly and were largely unaffected by
    sampling choice.
-   Naive Bayes showed moderate sensitivity to stratified and bootstrap
    sampling.
-   SVM was the weakest performer and degraded significantly under
    bootstrap sampling.
-   Tree-based and neural models were more robust to sampling changes
    than margin-based classifiers.

------------------------------------------------------------------------

##  Key Takeaways

-   Correct train-test splitting before resampling is critical to avoid
    inflated performance.
-   Sampling choice has limited impact on strong ensemble models.
-   Bootstrap sampling can harm sensitive models such as SVM.
-   Evaluations without leakage provide more realistic and trustworthy
    results.

------------------------------------------------------------------------

##  How to Run

``` bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
```

Open the notebook:

    102303007_Sampling.ipynb

Run all cells from top to bottom.

------------------------------------------------------------------------

