from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
import seaborn as sns
import matplotlib.pyplot as plt

# Results: (filename, number of detected objects)
results = [
    ('0.jpg', 0), ('112.jpg', 0), ('125.jpg', 0), ('137.jpg', 0), ('139.jpg', 0),
    ('142.jpg', 0), ('146.jpg', 0), ('154.jpg', 0), ('157.jpg', 0), ('158.jpg', 0),
    ('188.jpg', 0), ('191.jpg', 0), ('218.jpg', 0), ('230.jpg', 0), ('235.jpg', 0),
    ('246.jpg', 0), ('256.jpg', 0), ('263.jpg', 0), ('265.jpg', 0), ('266.jpg', 0),
    ('4.jpg', 0), ('47.jpg', 0), ('53.jpg', 0), ('55.jpg', 0), ('66.jpg', 0),
    ('Cars76.png', 0), ('Cars77.png', 1), ('Cars78.png', 1), ('Cars79.png', 0),
    ('Cars8.png', 2), ('Cars80.png', 1), ('Cars81.png', 0), ('Cars82.png', 1),
    ('Cars83.png', 1), ('Cars84.png', 0), ('Cars85.png', 0), ('Cars86.png', 1),
    ('Cars87.png', 0), ('Cars88.png', 1), ('Cars9.png', 1), ('Cars90.png', 0),
    ('Cars91.png', 0), ('Cars92.png', 0), ('Cars93.png', 0), ('Cars94.png', 0),
    ('Cars95.png', 0), ('Cars96.png', 1), ('Cars97.png', 0), ('Cars98.png', 0)
]

# Ground truth and predictions
y_true = [1 if 'Cars' in fname else 0 for fname, _ in results]  # Ground truth
y_pred = [1 if detected_objects > 0 else 0 for _, detected_objects in results]  # Predictions

# Metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

# Specificity: TN / (TN + FP)
specificity = tn / (tn + fp)

# Print metrics
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Specificity: {specificity:.2f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=['Negative', 'Positive']))

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
