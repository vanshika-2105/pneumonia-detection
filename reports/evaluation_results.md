
# Model Evaluation Results

## Dataset
- Test images: 624
- Normal images: 234
- Pneumonia images: 390

## Performance Metrics

| Metric | Score |
|---|---:|
| Accuracy | 79.33% |
| Precision | 75.74% |
| Recall | 98.46% |
| F1 Score | 85.62% |

## Confusion Matrix

| | Predicted Normal | Predicted Pneumonia |
|---|---:|---:|
| Actual Normal | 111 | 123 |
| Actual Pneumonia | 6 | 384 |

## Key Findings
- The model identifies pneumonia cases with high recall.
- The model struggles to identify normal X-rays.
- 123 normal images were incorrectly classified as pneumonia.
- Further model improvement is required.
