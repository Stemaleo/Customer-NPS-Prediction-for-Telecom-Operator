# Take-Home Challenge Write-Up: Customer NPS Prediction

## 1. Approach and Problem Framing
The objective of this challenge is to predict a customer's Net Promoter Score (NPS) category—Detractor, Passive, or Promoter—using their account and behavioral data. Because only 15% of the customer base typically answers NPS surveys, the business requires a machine learning system capable of inferring the NPS of the silent 85% majority.

This problem is framed as a **multi-class classification problem** with a severe class imbalance, operating under an ordinal constraint (Detractor < Passive < Promoter). 

## 2. Target Construction and Leakage Prevention
The provided dataset includes a `Satisfaction Score` ranging from 1 to 5. The NPS target was derived using the following strict mapping:
* **Promoter**: Satisfaction Score 5
* **Passive**: Satisfaction Score 4
* **Detractor**: Satisfaction Score <= 3

**Data Leakage Addressed:** 
A critical step in the data preparation phase was the identification and removal of "leakage" columns: `Churn Label`, `Churn Score`, `Churn Category`, and `Churn Reason`. If included, these columns would artificially inflate the model's performance, as a customer's decision to churn—and their stated reason for doing so—is a post-facto event that a predictive retention model would not have access to in a real-world scenario.

## 3. Data Splitting and Real-World Simulation
To accurately reflect the business reality where the model will be applied to the silent 85% of the base, the dataset was split with `train_size=0.15` and `test_size=0.85`. The training set simulates the survey respondents, while the massive test set acts as the silent majority. This deliberately makes the modeling task highly challenging but ensures that the evaluation metrics map directly to real-world expected performance.

## 4. Feature Engineering
A `scikit-learn` pipeline was constructed to seamlessly handle raw data:
* **Numeric Features** (e.g., `Tenure in Months`, `Monthly Charge`, `Total Charges`) were scaled using `StandardScaler` to ensure gradient-based and distance-based algorithms function optimally.
* **Categorical Features** (e.g., `Internet Type`, `Contract`, `Payment Method`) were transformed using `OneHotEncoder`.
* Demographics (`Senior Citizen`, `Gender`) were retained to allow for fairness auditing downstream.

## 5. Modeling Decisions and Evaluation
Given the imbalanced nature of the NPS classes and the minimal 15% training data, raw Accuracy is a fundamentally flawed metric. We optimized and evaluated the models based on **Macro-F1**, which equally weights the performance across all three classes, heavily penalizing models that ignore the rare "Detractor" class.

1. **Baseline Model:** A Logistic Regression model was trained using `class_weight='balanced'`. It provided a highly interpretable baseline with a Macro-F1 score of ~0.43.
2. **Advanced Model:** An XGBoost Classifier (`multi:softprob`) was implemented. To counter the class imbalance, custom sample weights were computed based on the inverse frequency of the classes in the 15% training set. The XGBoost model achieved a Macro-F1 of ~0.40. 

While the absolute F1 scores appear low, they accurately reflect the difficulty of extrapolating complex human sentiment from tabular billing data using only a 15% sample size.

## 6. Drivers of Detraction (Interpretability)
SHAP (SHapley Additive exPlanations) values were extracted using `TreeExplainer` on the XGBoost model to interpret the drivers of detraction. 
* **Actionable Levers:** Features such as `Monthly Charge`, `Internet Type_Fiber Optic`, and `Contract_Month-to-Month` are strong drivers of Detraction. If a predicted Detractor is identified, the retention team's most likely successful lever is transitioning the user from a Month-to-Month contract to a One Year contract, or offering a temporary discount to lower the Monthly Charge.
* **Non-Actionable Levers:** Tenure and demographics also play a role, but the business cannot change these. They serve strictly as targeting signals rather than intervention levers.

## 7. Fairness and Bias Audit
A model that predicts NPS can inadvertently dictate how retention budgets are allocated. We audited the model to ensure it does not systematically fail to identify Detractors in specific protected demographic groups.
* **Senior Citizens:** The model achieved a Detractor Recall of ~81% for Senior Citizens, compared to ~67% for non-seniors. 
* **Gender:** The model achieved a Detractor Recall of ~68% for Males and ~71% for Females.

**Conclusion on Fairness:** The model is highly effective at identifying older detractors. However, the Customer Experience team should be aware that the model is slightly less sensitive to younger detractors, which could lead to retention budgets skewing toward older demographics over time.

## 8. Limitations and Next Steps
* **Limitations:** The model relies heavily on tabular billing data, which lacks the nuance of direct customer interactions. Furthermore, the 15% training set is assumed to be a random sample; if angry customers are more likely to fill out surveys than passive ones, our training data is inherently biased (selection bias).
* **Next Steps:** The most valuable next step would be incorporating unstructured text data (e.g., customer support call transcripts or chat logs) using an LLM to extract sentiment features. This would provide a direct window into the customer's frustration levels, likely boosting the Macro-F1 score significantly.
