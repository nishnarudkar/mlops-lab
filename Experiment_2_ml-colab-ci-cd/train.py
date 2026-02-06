import joblib

from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score


# ---------------------------------------------------------
# BUILD STAGE (Train Linear Regression Model)
# ---------------------------------------------------------
def build():
    print("\n==============================")
    print(" BUILD Stage Started")
    print("==============================")

    # Load dataset for Linear Regression
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Linear Regression Model
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    print(" Linear Regression Model Training Completed!")

    return linear_model, X_test, y_test


# ---------------------------------------------------------
# TEST STAGE (Evaluate Linear Regression Model)
# ---------------------------------------------------------
def test(linear_model, X_test, y_test):
    print("\n==============================")
    print(" TEST Stage Started")
    print("==============================")

    # Predictions
    predictions = linear_model.predict(X_test)

    # Evaluation Metric
    mse = mean_squared_error(y_test, predictions)

    print(" Linear Regression Testing Completed!")
    print(" Mean Squared Error (MSE):", mse)


# ---------------------------------------------------------
# DEPLOY STAGE (Train + Save Logistic Regression Model)
# ---------------------------------------------------------
def deploy():
    print("\n==============================")
    print(" DEPLOY Stage Started")
    print("==============================")

    # Load dataset for Logistic Regression
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Feature Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train Logistic Regression Model
    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train, y_train)

    # Predictions
    y_pred = logistic_model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # Save the trained model
    joblib.dump(logistic_model, "logistic_model.pkl")

    print(" Logistic Regression Model Deployed Successfully!")
    print(" Accuracy:", accuracy)
    print(" Model Saved as logistic_model.pkl")


# ---------------------------------------------------------
# MAIN PIPELINE EXECUTION
# ---------------------------------------------------------
if __name__ == "__main__":
    print("\n Starting ML CI/CD Pipeline...")

    # Build Stage
    linear_model, X_test, y_test = build()

    # Test Stage
    test(linear_model, X_test, y_test)

    # Deploy Stage
    deploy()

    print("\n Pipeline Completed Successfully!")
