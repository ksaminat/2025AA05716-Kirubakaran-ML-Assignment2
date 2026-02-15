# ML Model Performance Report

This report summarizes the performance of all machine learning models used for diabetes prediction.

## Model Performance Table

| Model                    |   Accuracy |   Precision |   Recall |   F1 Score |      MCC |      AUC | Observation                                                                                                                                                                     |
|:-------------------------|-----------:|------------:|---------:|-----------:|---------:|---------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Logistic Regression      |      0.865 |    0.441176 | 0.114504 |   0.181818 | 0.17247  | 0.823734 | High accuracy and strong AUC, but very low recall means it misses many positive diabetes cases. Suitable for overall classification but not ideal for detecting positive cases. |
| Decision Tree            |      0.774 |    0.237569 | 0.328244 |   0.275641 | 0.148485 | 0.583948 | Lowest accuracy and AUC among all models, indicating weak performance. Moderate recall but unstable predictions due to overfitting and poor generalization.                     |
| kNN                      |      0.844 |    0.341772 | 0.206107 |   0.257143 | 0.182958 | 0.7239   | Good accuracy and moderate AUC, but low recall limits its ability to detect positive diabetes cases. Provides moderate performance overall.                                     |
| Naive Bayes              |      0.786 |    0.310502 | 0.519084 |   0.388571 | 0.281722 | 0.793287 | Best recall, F1-score, and MCC among all models. Most effective at detecting diabetes cases and provides balanced performance despite slightly lower accuracy.                  |
| Random Forest (Ensemble) |      0.865 |    0.46     | 0.175573 |   0.254144 | 0.223704 | 0.773975 | High accuracy and good MCC, showing reliable performance. However, low recall means it still misses many positive cases.                                                        |
| XGBoost (Ensemble)       |      0.867 |    0.478261 | 0.167939 |   0.248588 | 0.226003 | 0.816996 | Highest accuracy and precision among all models, with strong AUC. Provides best overall performance but recall is still relatively low.                                         |

## Overall Conclusion

- Best Accuracy: XGBoost (86.7%)
- Best Recall and F1 Score: Naive Bayes
- Best Balanced Performance: XGBoost and Random Forest
- Best for detecting diabetes cases: Naive Bayes
- Weakest Model: Decision Tree
