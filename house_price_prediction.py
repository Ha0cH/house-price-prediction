import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def main():
    print("Loading dataset...")
    
    # 1) Load dataset
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.copy()

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    # 2) Separate features and target
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    # 3) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4) Build pipeline
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])

    # 5) Train model
    print("\nTraining model...")
    model.fit(X_train, y_train)

    # 6) Predict
    predictions = model.predict(X_test)

    # 7) Evaluate
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    print(f"\nTest RMSE: {rmse:.4f}")

    # 8) Cross-validation
    print("\nRunning cross-validation...")
    cv_scores = -cross_val_score(
        model,
        X_train,
        y_train,
        scoring="neg_root_mean_squared_error",
        cv=5
    )

    print("Cross-validation RMSE scores:", cv_scores)
    print(f"Average CV RMSE: {cv_scores.mean():.4f}")

    # 9) Show predictions vs actual
    results = pd.DataFrame({
        "Actual": y_test.iloc[:10].values,
        "Predicted": predictions[:10]
    })

    print("\nSample predictions:")
    print(results)

    # 10) Feature importance (coefficients)
    coefficients = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.named_steps["regressor"].coef_
    }).sort_values(by="Coefficient", key=abs, ascending=False)

    print("\nFeature coefficients:")
    print(coefficients)

    print("\nDone.")

if __name__ == "__main__":
    main()