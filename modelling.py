import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
df = pd.read_csv('C:/Users/hp/Downloads/cleaned_Data.csv')

print("=" * 60)
print("CLASSIFICATION MODEL: Hit/Flop Movie Prediction")
print("=" * 60)

print("\nDefining Hit/Flop Movies: \n")

# Extract year
df['release_date'] = pd.to_datetime(df['release_date'])
df['release_year'] = df['release_date'].dt.year

# Calculate profit & profit ratio
df['profit'] = df['revenue'] - df['budget']
df['profit_ratio'] = df['profit'] / df['budget']

# Define Hit: Profit Ratio > 0.5
df['is_hit'] = (df['profit_ratio'] > 0.5).astype(int)

hits = df['is_hit'].sum()
flops = len(df) - hits
print(f"Hit movies: {hits} ({hits/len(df)*100:.1f}%)")
print(f"Flop movies: {flops} ({flops/len(df)*100:.1f}%)")

#Prepare Features
print("\nPreparing Features: \n")

features = ['budget', 'runtime', 'votes', 'popularity', 'release_year']
target = 'is_hit'

# Filter data
df_model = df.dropna(subset=features + [target])
df_model = df_model[df_model['budget'] > 0]

X = df_model[features]
y = df_model[target]

print(f"Total samples: {len(X)}")
print(f"Features: {features}")

print("\nTraining Model: \n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Train Random Forest
model = RandomForestClassifier(
    n_estimators=100, 
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)
print("\nRandom Forest Classifier trained")

#Evaluate Model
print("\nModel Evaluation:")

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_pred)
print(f"\nOverall Accuracy: {accuracy*100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Flop', 'Hit']))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(f"True Flops correctly predicted: {cm[0,0]}")
print(f"True Hits correctly predicted: {cm[1,1]}")
print(f"False positives (predicted Hit, was Flop): {cm[0,1]}")
print(f"False negatives (predicted Flop, was Hit): {cm[1,0]}")

#Calculating Cross-Validation
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Cross-validation accuracy: {scores.mean()*100:.2f}% (+/- {scores.std()*100:.2f}%)")

#Feature Importance
print("\nFeature Importance:")
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
print(feature_importance.to_string(index=False))

#Visualizations
print("\nGenerating visualizations\n")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Hit/Flop Movie Prediction Model', fontsize=16, fontweight='bold')

# Plot 1: Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['Flop', 'Hit'], yticklabels=['Flop', 'Hit'])
axes[0, 0].set_title(f'Confusion Matrix (Accuracy: {accuracy*100:.1f}%)')
axes[0, 0].set_ylabel('Actual')
axes[0, 0].set_xlabel('Predicted')

# Plot 2: Feature Importance
axes[0, 1].barh(feature_importance['Feature'], feature_importance['Importance'], color='teal')
axes[0, 1].set_xlabel('Importance')
axes[0, 1].set_title('Feature Importance')
axes[0, 1].invert_yaxis()

# Plot 3: Prediction Confidence Distribution
axes[1, 0].hist(y_pred_proba, bins=30, color='purple', alpha=0.7, edgecolor='black')
axes[1, 0].set_xlabel('Predicted Probability of Hit')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Model Confidence Distribution')
axes[1, 0].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
axes[1, 0].legend()

# Plot 4: Hit vs Flop by Budget
budget_bins = pd.cut(df_model['budget'], bins=5)
hit_rate = df_model.groupby(budget_bins)['is_hit'].mean()
hit_rate.plot(kind='bar', ax=axes[1, 1], color='orange', rot=45)
axes[1, 1].set_title('Hit Rate by Budget Range')
axes[1, 1].set_xlabel('Budget Range')
axes[1, 1].set_ylabel('Hit Rate')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hit_flop_prediction_model.png', dpi=300, bbox_inches='tight')
print("Saved visualization to 'hit_flop_prediction_model.png'")

