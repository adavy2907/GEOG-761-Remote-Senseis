import numpy as np
import os
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib
import datetime

# --- 1. Configuration and Data Loading ---
landslide_dir = "C:/Users/adavy/Downloads/landslides_data/landslide"
control_dir = "C:/Users/adavy/Downloads/landslides_data/nonlandslide"

# Define class labels explicitly
LANDSLIDE_LABEL = 1  # Landslide is POSITIVE class
NON_LANDSLIDE_LABEL = 0  # Non-landslide is NEGATIVE class

landslide_files = glob.glob(os.path.join(landslide_dir, "*.npy"))
control_files = glob.glob(os.path.join(control_dir, "*.npy"))

print("Found:")
print(f"  {len(landslide_files)} landslide files")
print(f"  {len(control_files)} control files")

# Quick check of one file
if landslide_files:
    sample = np.load(landslide_files[0])
    print("Sample shape:", sample.shape)

# --- 2. Feature extraction ---
def extract_features(img_array):
    """Extract mean, std, min, max per band."""
    features = []
    for band in range(img_array.shape[2]):
        b = img_array[:, :, band]
        features.extend([np.mean(b), np.std(b), np.min(b), np.max(b)])
    return features

# --- 3. Load data with explicit labeling ---
X, y = [], []

print("\nLoading data with explicit class labeling:")
print(f"  Landslide = {LANDSLIDE_LABEL} (POSITIVE class)")
print(f"  Non-landslide = {NON_LANDSLIDE_LABEL} (NEGATIVE class)")

for path in glob.glob(os.path.join(landslide_dir, "*.npy")):
    X.append(extract_features(np.load(path)))
    y.append(LANDSLIDE_LABEL)  # Landslide = POSITIVE

for path in glob.glob(os.path.join(control_dir, "*.npy")):
    X.append(extract_features(np.load(path)))
    y.append(NON_LANDSLIDE_LABEL)  # Non-landslide = NEGATIVE

X = np.array(X)
y = np.array(y)

print(f"Loaded {len(X)} samples with {X.shape[1]} features each.")
print(f"Class distribution: {np.bincount(y)} (Non-landslide: {np.sum(y==0)}, Landslide: {np.sum(y==1)})")

# --- 4. Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# --- 5. Train model with preprocessing pipeline ---
print("\nTraining model with RobustScaler preprocessing...")
pipeline = Pipeline([
    ('scaler', RobustScaler()),  # More robust to outliers than StandardScaler
    ('classifier', RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))
])

pipeline.fit(X_train, y_train)

# Get the classifier for feature importance
rf_classifier = pipeline.named_steps['classifier']

# Make predictions
y_pred = pipeline.predict(X_test)
y_prob_landslide = pipeline.predict_proba(X_test)[:, LANDSLIDE_LABEL]  # Probability of LANDSLIDE

# --- 6. Reports ---
print("\nClassification Report (Landslide as Positive Class):")
print(classification_report(y_test, y_pred, target_names=["Non-Landslide", "Landslide"]))

cm = confusion_matrix(y_test, y_pred)

# --- 7. Confusion matrix heatmap ---

# --- NEW SECTION: Weighted Average Performance Metrics Bar Chart ---
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Compute metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Prepare data
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
scores = [accuracy, precision, recall, f1]
colors = ['blue', 'green', 'gold', 'salmon']

# Plot
plt.figure(figsize=(6, 4))
bars = plt.bar(metrics, scores, color=colors)
plt.ylim(0, 1)
plt.ylabel('Score')
plt.title('Weighted Average Performance Metrics - Test Set', weight='bold')

# Annotate values
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width() / 2, score + 0.01,
             f"{score:.3f}", ha='center', va='bottom', fontsize=10, weight='bold')

plt.tight_layout()
plt.show()


def plot_confusion_matrix(y_true, y_pred_labels):
    from sklearn.metrics import confusion_matrix
    import numpy as np
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred_labels)
    labels = ["Non-Landslide", "Landslide"]

    # Compute row percentages
    row_sums = cm.sum(axis=1, keepdims=True)
    pct = np.round(cm / (row_sums + 1e-9) * 100, 1)

    # Combine counts and percentages
    annot = np.array([f"{count}\n({pc}%)" for count, pc in zip(cm.flatten(), pct.flatten())])
    annot = annot.reshape(cm.shape)

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = ax.imshow(cm, cmap="Blues", aspect="equal")

    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, annot[i, j], ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)

    # Axis labels and ticks
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix\n(Landslide = Positive Class)",
                 fontsize=14, fontweight="bold", pad=12)

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Number of Samples", fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    # Ensure full matrix and labels render properly
    ax.set_xlim(-0.5, cm.shape[1] - 0.5)
    ax.set_ylim(cm.shape[0] - 0.5, -0.5)

    plt.tight_layout()
    plt.show()
plot_confusion_matrix(y_test, y_pred)

# --- 8. Feature importances ---
feature_names = [f"{stat}_band{b+1}" for b in range(6) for stat in ["mean", "std", "min", "max"]]
importances = pd.Series(rf_classifier.feature_importances_, index=feature_names)
top_features = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 5))
sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")
plt.title("Top 10 Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# --- 9. ROC curve (Landslide as Positive) ---
fpr, tpr, _ = roc_curve(y_test, y_prob_landslide, pos_label=LANDSLIDE_LABEL)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})", color='blue')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Landslide Detection\n(Landslide = Positive Class)")
plt.legend()
plt.grid(True)
plt.show()

# --- 10. Precision–Recall curve (Landslide as Positive) ---
precision, recall, _ = precision_recall_curve(y_test, y_prob_landslide, pos_label=LANDSLIDE_LABEL)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, label="Precision-Recall Curve", color='red')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve - Landslide Detection\n(Landslide = Positive Class)")
plt.grid(True)
plt.legend()
plt.show()

# --- 11. Save trained model to Downloads folder ---
downloads_path = os.path.join(os.path.expanduser("~"), "Downloads")
os.makedirs(downloads_path, exist_ok=True)

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"rf_landslide_model_{timestamp}.pkl"
model_path = os.path.join(downloads_path, model_filename)

joblib.dump(pipeline, model_path)

print(f"\n✅ Model saved to: {model_path}")

# --- 12. Data Distribution Analysis ---
def compare_distributions(X_train, landslide_val_dir, nonlandslide_val_dir):
    """Compare feature distributions between training and validation data."""
    print("\n" + "="*60)
    print("DATA DISTRIBUTION ANALYSIS")
    print("="*60)
    
    # Extract features from validation data
    X_val_diag = []
    val_files = (glob.glob(os.path.join(landslide_val_dir, "*.npy")) + 
                 glob.glob(os.path.join(nonlandslide_val_dir, "*.npy")))
    
    for path in val_files:
        try:
            img_array = np.load(path)
            X_val_diag.append(extract_features(img_array))
        except Exception as e:
            continue
    
    if len(X_val_diag) == 0:
        print("❌ No validation data for diagnosis")
        return
    
    X_val_diag = np.array(X_val_diag)
    
    # Compare basic statistics
    print("\nTraining data statistics:")
    print(f"  Overall mean: {np.mean(X_train):.3f} ± {np.std(X_train):.3f}")
    print(f"  Value range: [{np.min(X_train):.3f}, {np.max(X_train):.3f}]")
    
    print("\nValidation data statistics:")
    print(f"  Overall mean: {np.mean(X_val_diag):.3f} ± {np.std(X_val_diag):.3f}")
    print(f"  Value range: [{np.min(X_val_diag):.3f}, {np.max(X_val_diag):.3f}]")
    
    # Check for dramatic differences
    train_mean = np.mean(X_train, axis=0)
    val_mean = np.mean(X_val_diag, axis=0)
    differences = np.abs(train_mean - val_mean) / (np.std(X_train, axis=0) + 1e-8)
    
    print(f"\nLargest normalized differences:")
    for i in np.argsort(differences)[-5:][::-1]:
        feature_name = feature_names[i] if i < len(feature_names) else f"Feature_{i}"
        print(f"  {feature_name}: {differences[i]:.2f}σ difference")
    
    return X_val_diag

# --- 13. Revised Validation Function ---
def validate_on_external_folder_corrected(pipeline, landslide_val_dir, nonlandslide_val_dir):
    """Validate the trained model on an external dataset with correct class labeling."""
    print("\n" + "="*60)
    print("VALIDATION ON EXTERNAL DATASET")
    print("="*60)
    print("Landslide = Positive Class (1), Non-Landslide = Negative Class (0)")
    
    # Load validation data
    X_val, y_val = [], []
    val_landslide_files = glob.glob(os.path.join(landslide_val_dir, "*.npy"))
    val_control_files = glob.glob(os.path.join(nonlandslide_val_dir, "*.npy"))
    
    print(f"Validation dataset:")
    print(f"  {len(val_landslide_files)} landslide files")
    print(f"  {len(val_control_files)} non-landslide files")
    
    # Process landslide validation samples (POSITIVE)
    landslide_count = 0
    for path in val_landslide_files:
        try:
            img_array = np.load(path)
            if img_array.shape[2] == 6:  # Check band consistency
                X_val.append(extract_features(img_array))
                y_val.append(LANDSLIDE_LABEL)
                landslide_count += 1
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    # Process non-landslide validation samples (NEGATIVE)
    nonlandslide_count = 0
    for path in val_control_files:
        try:
            img_array = np.load(path)
            if img_array.shape[2] == 6:
                X_val.append(extract_features(img_array))
                y_val.append(NON_LANDSLIDE_LABEL)
                nonlandslide_count += 1
        except Exception as e:
            print(f"Error loading {path}: {e}")
    
    print(f"Successfully loaded: {landslide_count} landslide, {nonlandslide_count} non-landslide")
    
    if len(X_val) == 0:
        print("❌ No validation data loaded. Check file paths and formats.")
        return
    
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    
    print(f"Loaded {len(X_val)} validation samples")
    print(f"Validation class distribution: {np.bincount(y_val)}")
    
    # Make predictions using pipeline (includes scaling)
    y_val_pred = pipeline.predict(X_val)
    y_val_prob_landslide = pipeline.predict_proba(X_val)[:, LANDSLIDE_LABEL]  # Probability of LANDSLIDE
    
    # Validation reports
    print("\n📊 Validation Classification Report:")
    print(classification_report(y_val, y_val_pred, target_names=["Non-Landslide", "Landslide"]))
    
    # Validation confusion matrix
    cm_val = confusion_matrix(y_val, y_val_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm_val, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Non-Landslide", "Landslide"],
                yticklabels=["Non-Landslide", "Landslide"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Validation Confusion Matrix\n(Landslide = Positive Class)")
    plt.show()
    
    # Validation ROC curve
    fpr_val, tpr_val, _ = roc_curve(y_val, y_val_prob_landslide, pos_label=LANDSLIDE_LABEL)
    roc_auc_val = auc(fpr_val, tpr_val)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr_val, tpr_val, label=f"Validation AUC = {roc_auc_val:.3f}", color='red', linewidth=2)
    # Compare with original test ROC
    plt.plot(fpr, tpr, label=f"Original Test AUC = {roc_auc:.3f}", color='blue', linestyle='--')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve: Original Test vs Validation\n(Landslide = Positive Class)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Validation Precision-Recall curve
    precision_val, recall_val, _ = precision_recall_curve(y_val, y_val_prob_landslide, pos_label=LANDSLIDE_LABEL)
    
    plt.figure(figsize=(6, 5))
    plt.plot(recall_val, precision_val, label="Validation PR Curve", color='red', linewidth=2)
    # Compare with original test PR curve
    plt.plot(recall, precision, label="Original Test PR Curve", color='blue', linestyle='--')
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve: Original Test vs Validation\n(Landslide = Positive Class)")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Model behavior analysis
    print(f"\n🔍 Model Behavior Analysis:")
    print(f"  Landslide predictions: {np.sum(y_val_pred == LANDSLIDE_LABEL)}/{len(y_val_pred)}")
    print(f"  Average landslide probability: {np.mean(y_val_prob_landslide):.3f}")
    print(f"  Probability range: [{np.min(y_val_prob_landslide):.3f}, {np.max(y_val_prob_landslide):.3f}]")
    
    # Calculate and display accuracy
    val_accuracy = np.mean(y_val_pred == y_val)
    print(f"\n✅ Validation Accuracy: {val_accuracy:.4f}")
    
    return {
        'X_val': X_val,
        'y_val': y_val,
        'y_val_pred': y_val_pred,
        'y_val_prob_landslide': y_val_prob_landslide,
        'accuracy': val_accuracy,
        'roc_auc': roc_auc_val
    }

# --- 14. Run Validation ---
landslide_val_dir = "C:/Users/adavy/Downloads/validation/landslide"
nonlandslide_val_dir = "C:/Users/adavy/Downloads/validation/nonlandslide"

# Check if validation directories exist
if os.path.exists(landslide_val_dir) and os.path.exists(nonlandslide_val_dir):
    # First run data distribution analysis
    X_val_diagnostic = compare_distributions(X_train, landslide_val_dir, nonlandslide_val_dir)
    
    # Then run validation
    validation_results = validate_on_external_folder_corrected(pipeline, landslide_val_dir, nonlandslide_val_dir)
else:
    print(f"\n⚠️  Validation directories not found:")
    print(f"   Landslide: {landslide_val_dir}")
    print(f"   Non-landslide: {nonlandslide_val_dir}")
    print("Please update the directory paths in the code.")

