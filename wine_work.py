from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

import mlflow
from mlflow.models import infer_signature

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier

import kagglehub
from kagglehub import KaggleDatasetAdapter
import matplotlib.pyplot as plt

file_path = "wine_quality_classification.csv"


df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "sahideseker/wine-quality-classification",
  file_path,
  
)


quality_order = ["low", "medium", "high"]  
encoder = OrdinalEncoder(
    categories=[quality_order],
    handle_unknown='use_encoded_value',  
    unknown_value=-1  
)
y_encoded = encoder.fit_transform(df[['quality_label']]).ravel()


X = df.drop(columns="quality_label")
y = y_encoded
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  
)



params_lr = {
    "solver": "lbfgs",
    "max_iter": 10000,
    "random_state": 8888,

    "class_weight": "balanced",  
    "penalty": "l2",
    "C": 0.1  
}

lr = LogisticRegression(**params_lr)
lr.fit(X_train, y_train)


y_pred_lr = lr.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_lr)



y_proba = lr.predict_proba(X_test)
report_dict_lr = classification_report(y_test, y_pred_lr, output_dict=True)



params_rf = {
    "n_estimators": 30,
    "max_depth": 3
}
rf_clf = RandomForestClassifier(**params_rf)
rf_clf.fit(X_train, y_train)
y_pred_rf = rf_clf.predict(X_test)
report_dict_rf = classification_report(y_test, y_pred_rf, output_dict=True)



models = {"LogisticRegression": lr, "RandomForest": rf_clf}
params = {"LogisticRegression": params_lr, "RandomForest": params_rf}
report_dict = {"LogisticRegression": report_dict_lr, "RandomForest": report_dict_rf}
wine_feature_names = list(X_train.columns)




mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
mlflow.set_experiment("2-MLflow_Wine_Nested")

with mlflow.start_run(run_name="Wine_Model_Comparison"):
    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name, nested=True):
            mlflow.log_params(params[model_name])
            mlflow.log_metrics({
                "accuracy": report_dict[model_name]["accuracy"],
                "recall_class_0": report_dict[model_name]["0.0"]["recall"],
                "recall_class_1": report_dict[model_name]["1.0"]["recall"],
                "recall_class_2": report_dict[model_name]["2.0"]["recall"],
                "f1-score": report_dict[model_name]["macro avg"]["f1-score"]
            })
            mlflow.set_tag("Training Info", f"{model_name} model for Wine")
            signature = infer_signature(X_train, model.predict(X_train))
            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=f"{model_name.lower()}_model",
                signature=signature,
                input_example=X_train,
                registered_model_name=f"tracking-wine-{model_name.lower()}"
            )
            predictions = model.predict(X_test)
            result = pd.DataFrame(X_test, columns=wine_feature_names).drop(columns="quality_label", errors="ignore")
            result["actual_class"] = y_test
            result["predicted_class"] = predictions
            result.to_csv(f"{model_name}_predictions.csv")
            mlflow.log_artifact(f"{model_name}_predictions.csv")
            
            
            
            plt.figure()
        if model_name == "LogisticRegression":
            
            coef = model.coef_[0]  
            plt.bar(wine_feature_names, coef)
            plt.title("LogisticRegression Coefficients")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig("logistic_coefficients.png")
            mlflow.log_artifact("logistic_coefficients.png")
        elif model_name == "RandomForest":
            
            importances = model.feature_importances_
            plt.bar(wine_feature_names, importances)
            plt.title("RandomForest Feature Importance")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig("feature_importance.png")
            mlflow.log_artifact("feature_importance.png")
        plt.close()