import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv("C:\\Users\\91858\\OneDrive\\Documents\\Desktop\\city_day.csv")

print(f"[INFO] Original dataset shape: {df.shape}")
print(f"[INFO] Columns: {list(df.columns)}\n")


# -------------------------
# Auto Target Detection
# -------------------------
def detect_target(df):
    possible_targets = [
        "target", "label", "diagnosis", "disease", "Outcome",
        "Liver_Disease", "LiverDisease", "has_disease",
        "AQI_Bucket", "AQI", "Category"
    ]
    for col in df.columns:
        if col.lower() in [t.lower() for t in possible_targets]:
            print(f"[INFO] Auto-detected target column: {col}")
            return col
    return None


# -------------------------
# Synthetic Target Creation
# -------------------------
def create_synthetic_target(df):
    print("[INFO] No target found. Creating synthetic target based on numeric median...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df["SyntheticTarget"] = (df[numeric_cols].mean(axis=1) >
                             df[numeric_cols].mean().mean()).astype(int)
    return "SyntheticTarget"


# -------------------------
# Enhanced EDA (fixed all warnings)
# -------------------------
def perform_eda(df):
    print("\n🔍 ----- Exploratory Data Analysis ----- 🔍")
    print("\nDataset Info:")
    print(df.info())

    print("\nSummary Statistics:")
    print(df.describe())

    print("\nMissing Values per Column:")
    print(df.isnull().sum())

    # Correlation Heatmap
    plt.figure(figsize=(10, 6))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    # AQI Distribution
    if "AQI" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df["AQI"], kde=True, bins=30, color="seagreen")
        plt.title("Distribution of AQI Levels")
        plt.xlabel("Air Quality Index (AQI)")
        plt.ylabel("Frequency")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

    # AQI Trend Over Time (if Date column exists)
    date_cols = [col for col in df.columns if "date" in col.lower()]
    if date_cols:
        date_col = date_cols[0]
        df.loc[:, date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df_sorted = df.sort_values(by=date_col)
        if "AQI" in df.columns:
            plt.figure(figsize=(12, 5))
            plt.plot(df_sorted[date_col], df_sorted["AQI"], color="purple", linewidth=1.5)
            plt.title(f"AQI Trend Over Time ({date_col})")
            plt.xlabel("Date")
            plt.ylabel("AQI")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

    # AQI Bucket Count (fixed seaborn warning)
    if "AQI_Bucket" in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(
            data=df,
            x="AQI_Bucket",
            hue="AQI_Bucket",
            order=df["AQI_Bucket"].value_counts().index,
            palette="viridis",
            legend=False
        )
        plt.title("Count of Each AQI Bucket")
        plt.xlabel("AQI Bucket Category")
        plt.ylabel("Number of Days")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Outlier Detection
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:5]
    if len(numeric_cols) > 0:
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=df[numeric_cols])
        plt.title("Outlier Detection in Numeric Features")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Pairplot
    if len(numeric_cols) >= 2:
        sns.pairplot(df[numeric_cols], diag_kind="kde", corner=True)
        plt.suptitle("Pairplot of Selected Numeric Features", y=1.02)
        plt.show()

    print("\n✅ EDA Completed Successfully!")


# -------------------------
# Preprocessing (fixed datetime issue)
# -------------------------
def preprocess_data(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Drop any datetime columns before training
    datetime_cols = X.select_dtypes(include=["datetime64", "datetime64[ns]"]).columns
    if len(datetime_cols) > 0:
        print(f"[INFO] Dropping datetime columns: {list(datetime_cols)}")
        X = X.drop(columns=datetime_cols)

    # Convert categorical variables
    X = pd.get_dummies(X, drop_first=True)

    # Standardize numeric data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, X.columns


# -------------------------
# Model Training & Evaluation
# -------------------------
def run_model(X_train, X_test, y_train, y_test, feature_names, top_n=12):
    model = RandomForestClassifier(random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print("\n✅ Model Evaluation Report:")
    print(classification_report(y_test, y_pred))
    print(f"Overall Accuracy: {acc:.3f}\n")

    # Feature Importances
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top_features = importances.sort_values(ascending=False).head(top_n)
    print(f"Top {top_n} Important Features:")
    print(top_features)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x=top_features.values,
        y=top_features.index,
        hue=top_features.index,
        palette="viridis",
        legend=False
    )
    plt.title(f"Top {top_n} Feature Importances (Random Forest)")
    plt.xlabel("Importance")
    plt.ylabel("Feature Name")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    return model, acc


# -------------------------
# Save Model
# -------------------------
def save_model(model, filename="trained_rf_model.pkl"):
    joblib.dump(model, filename)
    print(f"[INFO] Model saved successfully as {filename}")


# -------------------------
# Main
# -------------------------
def main():
    # Drop rows with missing values
    df_clean = df.dropna(axis=0, how='any')
    print(f"[INFO] Dataset shape after cleaning: {df_clean.shape}")

    perform_eda(df_clean)

    target_col = detect_target(df_clean)
    if target_col is None:
        target_col = create_synthetic_target(df_clean)

    X, y, feature_names = preprocess_data(df_clean, target_col)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    model, acc = run_model(X_train, X_test, y_train, y_test, feature_names, top_n=12)
    save_model(model)

    print(f"\n🏁 Final Model Accuracy: {acc:.3f}")
    print("[INFO] Pipeline execution complete!")


if __name__ == "__main__":
    main()
