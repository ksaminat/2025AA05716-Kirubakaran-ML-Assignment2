# ML Model Performance Report

## Model Performance Metrics

| ML Model Name            |   Accuracy |   Precision |   Recall |   F1 Score |      MCC |      AUC |
|:-------------------------|-----------:|------------:|---------:|-----------:|---------:|---------:|
| Logistic Regression      |      0.865 |    0.441176 | 0.114504 |   0.181818 | 0.17247  | 0.823734 |
| Decision Tree            |      0.774 |    0.237569 | 0.328244 |   0.275641 | 0.148485 | 0.583948 |
| kNN                      |      0.844 |    0.341772 | 0.206107 |   0.257143 | 0.182958 | 0.7239   |
| Naive Bayes              |      0.786 |    0.310502 | 0.519084 |   0.388571 | 0.281722 | 0.793287 |
| Random Forest (Ensemble) |      0.865 |    0.46     | 0.175573 |   0.254144 | 0.223704 | 0.773975 |
| XGBoost (Ensemble)       |      0.867 |    0.478261 | 0.167939 |   0.248588 | 0.226003 | 0.816996 |

## Model Performance Observations

| ML Model Name            | Observation about model performance                                                                                                                                                                                                                                                                                                                                                   |
|:-------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Logistic Regression      | Logistic Regression achieved high accuracy (86.5%) and strong AUC (0.824), indicating good overall classification capability. However, it has very low recall (11.45%), meaning it fails to identify many positive diabetes cases. This suggests the model is biased toward the majority class and is not ideal when detecting positive cases is critical.                            |
| Decision Tree            | Decision Tree showed lower accuracy (77.4%) and the lowest AUC (0.584), indicating weak overall performance. While recall (32.82%) is better than Logistic Regression, its precision and MCC are relatively low, suggesting unstable predictions and higher misclassification. This model tends to overfit and does not generalize well.                                              |
| kNN                      | kNN achieved good accuracy (84.4%) and moderate AUC (0.724). However, recall (20.61%) and F1 score are relatively low, meaning it struggles to detect positive cases effectively. Overall, it performs moderately well but is not optimal for imbalanced datasets like diabetes prediction.                                                                                           |
| Naive Bayes              | Naive Bayes has moderate accuracy (78.6%) but the highest recall (51.91%) and highest F1 score (0.389), indicating it is best at detecting positive diabetes cases among all models. It also has the highest MCC (0.282), showing better balanced performance. This model is highly suitable when identifying positive cases is important.                                            |
| Random Forest (Ensemble) | Random Forest achieved high accuracy (86.5%) and good AUC (0.774), showing strong overall performance. It has better precision and MCC than most models, indicating more reliable predictions. However, recall (17.56%) is still low, meaning it misses many positive cases despite strong overall classification.                                                                    |
| XGBoost (Ensemble)       | XGBoost achieved the highest accuracy (86.7%) and strong AUC (0.817), indicating excellent overall classification performance. It also has the highest precision (47.83%), meaning fewer false positives. However, recall (16.79%) is relatively low, so it may miss some positive cases. Overall, it provides the best balance between accuracy and precision among ensemble models. |

## Overall Conclusion

- Best Accuracy: XGBoost (86.7%)
- Best Recall and F1 Score: Naive Bayes
- Best Balanced Performance: XGBoost and Random Forest
- Best for detecting diabetes cases: Naive Bayes
- Weakest Model: Decision Tree
