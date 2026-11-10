
# Original vs Improved CNN Model Comparison

## Experiment
The original CNN was compared with a class-weighted CNN.

Both models were evaluated on the same 624 test images.

## Performance

| Metric | Original Model | Improved Model |
|---|---:|---:|
| Accuracy | 79.33% | 77.40% |
| Precision | 75.74% | 73.99% |
| Pneumonia Recall | 98.46% | 98.46% |
| F1 Score | 85.62% | 84.49% |

## Confusion Matrix Comparison

### Original Model
- True Normal: 111
- False Pneumonia: 123
- Missed Pneumonia: 6
- True Pneumonia: 384

### Improved Model
- True Normal: 99
- False Pneumonia: 135
- Missed Pneumonia: 6
- True Pneumonia: 384

## Conclusion

Class weighting did not improve the test performance.

The original model achieved higher accuracy, precision, and F1-score.

Both models achieved the same pneumonia recall.

Further experiments are required to improve normal image classification.
