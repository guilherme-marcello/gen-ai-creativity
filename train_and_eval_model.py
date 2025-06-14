import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    cross_validate,
    StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score
)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

# === Load the dataset ===
df = pd.read_csv("creativity_classification_dataset.csv")

# === Features & target ===
X = df[["concepts_distance", "novelty_score", "coherence_score", "emergence_score"]]
y = df["human_generated"]

# === Stratified K-Fold for balanced splits ===
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# === Create pipeline ===
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(random_state=42))
])

# === Cross-validate ===
cv_results = cross_validate(
    pipeline,
    X,
    y,
    cv=cv,
    scoring=["accuracy", "roc_auc", "f1"],
    return_train_score=True
)

# === Report cross-val metrics ===
print("\n✅ Cross-Validation Scores (5-fold):")
for metric in ["train_accuracy", "test_accuracy", "test_roc_auc", "test_f1"]:
    scores = cv_results[metric]
    print(f"{metric}: {scores.mean():.3f} ± {scores.std():.3f}")

# === Train/Test split for final eval and visualization ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Final model fit & test evaluation ===
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

print("\n✅ Final Holdout Test Report:")
print(classification_report(y_test, y_pred, digits=3))
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"ROC AUC:   {roc_auc_score(y_test, y_prob):.3f}")

### Confusion Matrix Plot

# === Confusion matrix ===
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Machine", "Human"], yticklabels=["Machine", "Human"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Holdout Test)")
plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight") # Save
#plt.show()

# === Perform PCA with 3 components ===
X_scaled = pipeline.named_steps["scaler"].transform(X)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

# === Prepare dataframe with PCA components and labels ===
pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2", "PC3"])
pca_df["actual_label"] = y.values
pca_df["predicted_label"] = pipeline.predict(X)

# === Map for visualization ===
colors = {0: "red", 1: "blue"}
markers = {0: "o", 1: "s"}

# === 3D Plot ===
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection="3d")

for label in pca_df["actual_label"].unique():
    for pred in pca_df["predicted_label"].unique():
        subset = pca_df[(pca_df["actual_label"] == label) & (pca_df["predicted_label"] == pred)]
        ax.scatter(
            subset["PC1"],
            subset["PC2"],
            subset["PC3"],
            c=colors[label],
            marker=markers[pred],
            edgecolor="black",
            label=f"Actual: {'Human' if label else 'Machine'}, Pred: {'Human' if pred else 'Machine'}",
            alpha=0.8,
            s=80
        )

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
ax.set_title("3D PCA Projection of Creativity Features")
ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1), borderaxespad=0.)
plt.tight_layout()
plt.savefig("3d_pca_projection.png", dpi=300, bbox_inches="tight") # Save
#plt.show()