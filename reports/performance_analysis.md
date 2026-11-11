# Day 9 - Model Performance Analysis

## Overall Accuracy
79.33%

## Confusion Matrix

| Metric | Count |
|---|---:|
| True Negatives (TN) | 111 |
| False Positives (FP) | 123 |
| False Negatives (FN) | 6 |
| True Positives (TP) | 384 |

## Classification Report

```text
              precision    recall  f1-score   support

      NORMAL     0.9487    0.4744    0.6325       234
   PNEUMONIA     0.7574    0.9846    0.8562       390

    accuracy                         0.7933       624
   macro avg     0.8531    0.7295    0.7443       624
weighted avg     0.8291    0.7933    0.7723       624

```

## Error Analysis

- False Positives: 123 normal images were incorrectly predicted as pneumonia.
- False Negatives: 6 pneumonia images were incorrectly predicted as normal.
- True Negatives: 111 normal images were correctly identified.
- True Positives: 384 pneumonia images were correctly identified.

## Conclusion

The model identifies pneumonia cases with high recall, but it also produces
false positive predictions for some normal images. This means further
validation would be necessary before any real-world medical use.

**Note:** These results describe performance on this test dataset and do
not establish clinical reliability.
