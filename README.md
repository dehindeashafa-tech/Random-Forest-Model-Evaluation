# README: Random-Forest-Model-Evaluation
# Random Forest Classification – Airline Customer Satisfaction Analysis


This notebook details the process of building and optimizing a Random Forest classification model to predict airline customer satisfaction using the `Invistico_Airline.csv` dataset. The analysis covers data preprocessing, model training, hyperparameter tuning, and a comparative evaluation against a Decision Tree model.

## 1. Dataset Overview

The `Invistico_Airline.csv` dataset contains passenger survey data, including various features related to travel experience and a target variable indicating 'satisfaction' (satisfied/dissatisfied). Key characteristics of the dataset:

*   **Total Entries**: 129,880
*   **Features**: 21 (e.g., Age, Flight Distance, Seat comfort, Inflight entertainment, Customer Type, Type of Travel, Class, Departure/Arrival Delay in Minutes, etc.)
*   **Target Variable**: `satisfaction` (categorical, encoded to 1 for 'satisfied' and 0 for 'dissatisfied')
*   **Missing Values Handled**: 'Arrival Delay in Minutes' were imputed with 0, assuming no delay for missing entries.

## 2. Hyperparameter Tuning Approach

To optimize the performance of the `RandomForestClassifier`, `GridSearchCV` was employed. This method systematically explores a predefined set of hyperparameter combinations, using cross-validation to find the combination that yields the best model performance.

### Parameters Tuned:

The `param_grid` for `GridSearchCV` included the following hyperparameters for the `RandomForestClassifier`:

*   `n_estimators`: [100, 200, 300] (Number of trees in the forest)
*   `max_depth`: [10, 20, 30, None] (Maximum depth of individual trees)
*   `min_samples_leaf`: [1, 2, 4] (Minimum number of samples required to be at a leaf node)

The optimization metric used for `GridSearchCV` was `roc_auc`, which is suitable for binary classification tasks and accounts for class imbalance.

## 3. Best Parameters Found

After running `GridSearchCV`, the following optimal hyperparameters were identified for the `RandomForestClassifier`:

*   `classifier__max_depth`: `30`
*   `classifier__min_samples_leaf`: `1`
*   `classifier__n_estimators`: `300`

These parameters yielded the best cross-validation AUC-ROC score of **0.9916** on the training data.

## 4. Side-by-Side Model Comparison (on Validation Set)

To assess the impact of using an ensemble method (Random Forest) and hyperparameter tuning, we compared the performance of three models on the validation set:

*   **Decision Tree**: A single Decision Tree Classifier with default parameters.
*   **Initial Random Forest**: A `RandomForestClassifier` with default parameters.
*   **Tuned Random Forest**: The `RandomForestClassifier` with the best hyperparameters found by `GridSearchCV`.

| Model                   | Accuracy | Precision | Recall   | F1-Score | AUC-ROC  |
| :---------------------- | :------- | :-------- | :------- | :------- | :------- |
| Decision Tree           | 0.9319   | 0.9342    | 0.9420   | 0.9381   | 0.9309   |
| Initial Random Forest   | 0.9567   | 0.9677    | 0.9526   | 0.9601   | 0.9931   |
| Tuned Random Forest     | 0.9577   | 0.9679    | 0.9544   | 0.9611   | 0.9933   |

### F1-Score Explained:

The F1-score is the harmonic mean of Precision and Recall. It is particularly useful when you have an uneven class distribution, as it punishes models that predict too many false positives or false negatives. A high F1-score indicates that the model has good precision and recall.

$F1 = 2 * (Precision * Recall) / (Precision + Recall)$

### Key Takeaways from Comparison:

*   **Random Forest Advantage**: Both the initial and tuned Random Forest models significantly outperformed the standalone Decision Tree across all metrics, highlighting the benefits of ensemble learning in handling complexity and improving generalization.
*   **Impact of Tuning**: While the initial Random Forest already performed strongly, hyperparameter tuning led to marginal but consistent improvements in all metrics, confirming the effectiveness of the optimization process.
*   **Overfitting Reduction**: Random Forest models are inherently less prone to overfitting compared to single Decision Trees due to their ensemble nature, making them more robust for predictive tasks.

## 5. Final Model Performance on Test Set

The best-tuned Random Forest model was evaluated on a completely unseen test set to provide an unbiased estimate of its real-world performance:

*   **Accuracy**: 0.9558
*   **Precision**: 0.9682
*   **Recall**: 0.9506
*   **F1-Score**: 0.9593
*   **AUC-ROC**: 0.9927

These results confirm the strong generalization capabilities of the optimized Random Forest model.

## 6. Key Satisfaction Drivers

Based on feature importance analysis from the tuned Random Forest model, the most influential factors driving customer satisfaction are:

1.  **Inflight entertainment**: The most significant factor, indicating that positive experiences here greatly increase satisfaction.
2.  **Seat comfort**: A highly important physical aspect of the flight experience.
3.  **Ease of Online booking**: The convenience and simplicity of the online booking process.
4.  **Online support**: Readily available and effective online support.
5.  **Food and drink**: The quality and variety of food and beverage offerings.
