import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

# Load data
ASD_data = pd.read_csv("ASD_meta_abundance.csv")
OTU_data = pd.read_csv("GSE113690_Autism_16S_rRNA_OTU_assignment_and_abundance.csv")

# Data Cleaning: Drop or impute missing values
# Exclude non-numeric columns from median calculation
ASD_data_numeric = ASD_data.select_dtypes(include=[np.number])
ASD_data[ASD_data_numeric.columns] = ASD_data_numeric.fillna(ASD_data_numeric.median())

OTU_data_numeric = OTU_data.select_dtypes(include=[np.number])
OTU_data[OTU_data_numeric.columns] = OTU_data_numeric.fillna(OTU_data_numeric.median())

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 8))
sns.heatmap(ASD_data_numeric.corr(), annot=False, cmap='coolwarm')
plt.title("Correlation Heatmap - ASD Data (Numeric Columns)")
plt.show()

# PCA to reduce dimensionality and visualize variance
pca = PCA(n_components=2)
ASD_pca = pca.fit_transform(StandardScaler().fit_transform(ASD_data_numeric))
plt.scatter(ASD_pca[:, 0], ASD_pca[:, 1], alpha=0.5)
plt.title("PCA of ASD Data (Numeric Columns)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

# Preparing data for modeling
X = ASD_data_numeric.iloc[:, :-1]
y = ASD_data.iloc[:, -1]

# Handle class imbalance by removing classes with very few samples
y_counts = y.value_counts()
valid_classes = y_counts[y_counts > 1].index
y = y[y.isin(valid_classes)]
X = X.loc[y.index]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardizing features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models and hyperparameter tuning
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier()
}

param_grid = {
    'Logistic Regression': {'C': [0.1, 1, 10]},
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
    'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    'KNN': {'n_neighbors': [3, 5, 7]}
}

for name, model in models.items():
    grid_search = GridSearchCV(model, param_grid[name], cv=5, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_

    # Predict on the test set
    y_pred = best_model.predict(X_test_scaled)

    # Print results
    print(f"Model: {name}")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))

# Feature Selection using RFE with Logistic Regression
rfe = RFE(LogisticRegression(), n_features_to_select=5)
rfe.fit(X_train_scaled, y_train)
print("Selected Features by RFE:", X.columns[rfe.support_])