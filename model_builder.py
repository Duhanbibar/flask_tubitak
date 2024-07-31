from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from joblib import dump
from engine import engine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,precision_score,recall_score,f1_score
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error,r2_score
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pickle_needs import *
from loader import loader,get_models
import random
import string
from sklearn.model_selection import GroupKFold,StratifiedGroupKFold
from metric_calculations import *

"""
Model Builder
Model Builder script builds models and evaluates them.
After the evaluation, models are collected in the format below and this dictionary is saved through joblib. 

    data = {
        "model":(Object),
        "name": (String),
        "metrics": (evaluation dictionary with metric names as keys and metrics as values),
        "id": "an unique id for the created model for using as an id in html files"
    }

Below can be seen some example model builds, One only needs to call preprocessor pipeline and transform the data into the expected format, then, build an estimator like in the example below:

        X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2)
        y_train_values = y_train["energy_value"]
        y_test_values = y_test["energy_value"]
        y_train_labels = y_train["energy_class"]
        y_test_labels = y_test["energy_class"]

        ### Random Forests Classification
        from sklearn.ensemble import RandomForestClassifier
        rfc = RandomForestClassifier()
        rfc.fit(X_train,y_train_labels)
        save_classifier_and_metrics_report(rfc,X_test,y_test_labels)

        (save_regressor_and_metrics_report shall be used in regression models.)

        In cases where further encoding is needed (for example, SVC classifier needs float labels instead of strings),
        model can be a Pipeline object. See example SVC model below.


"""


cv = False
log_file = "logs-model-builder.txt"


regressor_counter = 0
classifier_counter = 0

save_model_owner = "user"

def save_regressor_and_metrics_report(model,X_test,y_test,model_name=None):
    global regressor_counter,save_model_owner
    id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))

    model,regression_eval = regressor_report(model,X_test,y_test)

    if not model_name:
        model_name = type(model).__name__
    data = {
        "model":model,
        "name": model_name,
        "metrics": regression_eval,
        "id": id,
        "path":f"regressors/{save_model_owner}/{model_name}.joblib"
    }
    regressor_counter +=1
    print(f"{data['name']}{data['metrics']['r2']}")

    dump(data,f"regressors/{save_model_owner}/{model_name}.joblib")


def save_classifier_and_metrics_report(model,X_test,y_test,model_name=None):
    global classifier_counter,save_model_owner
    model,classification_metrics = get_classification_metrics(model,X_test,y_test)

    if not model_name:
        model_name = type(model).__name__

    id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    data = {
        "model":model,
        "name": model_name,
        "metrics": classification_metrics,
        "id": id,
        "path":f"classifiers/{save_model_owner}/{model_name}.joblib"
    }
    print(f"{data['name']}{data['metrics']['f1']}")
    classifier_counter += 1

    dump(data,f"classifiers/{save_model_owner}/{model_name}.joblib")


def train_classifiers():
    df = engine.get_table(as_df=True)
    df["product_code"] = df["product_code"].astype(str)
    X = df.drop(["energy_value","energy_class"],axis=1)
    X = loader.load_preprocessor().transform(X)
    Y = df["energy_class"]


    ### Random Forests Classification
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier()
    save_classifier_and_metrics_report(rfc,X,Y,"RF Sınıflandırıcı")
    ### Support Vector Classification

    from sklearn.svm import SVC
    svc_with_label_encoding = SVCWithLabelEncoding(SVC(kernel="sigmoid")) # Sigmoid fena değil
    svc_pipeline = Pipeline(steps=[
        ('svc_with_label_encoding', svc_with_label_encoding)
    ])
    save_classifier_and_metrics_report(svc_pipeline,X,Y,"SVM Sınıflandırıcı")


    ### Logiistic Regression
    from sklearn.linear_model import LogisticRegression
    logi = LogisticRegression(max_iter = 100000)
    save_classifier_and_metrics_report(logi,X,Y,"Lojistik Regresyon")

    from xgboost import XGBClassifier
    xgbc = XGBClassifier()
    xgbc_with_label_encoding = XGBCWithLabelEncoding(xgbc)
    xgbc_pipeline = Pipeline(steps=[
        ('xgbc_with_label_encoding', xgbc_with_label_encoding)
    ])
    save_classifier_and_metrics_report(xgbc_pipeline,X,Y,"XGBoost Sınıflandırıcı")

def train_regressors():
    df = engine.get_table(as_df=True)
    df["product_code"] = df["product_code"].astype(str)
    X = df.drop(["energy_value","energy_class"],axis=1)
    X = loader.load_preprocessor().transform(X)
    Y = df["energy_value"]



    ### Random Forests Regression
    from sklearn.ensemble import RandomForestRegressor
    rfr = RandomForestRegressor(max_depth= 20, min_samples_leaf= 4, min_samples_split= 2, n_estimators= 50)
    save_regressor_and_metrics_report(rfr,X,Y,"RF Regresyon")
    ### Linear Regression
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    save_regressor_and_metrics_report(lr,X,Y,"Doğrusal Regresyon")

    ### Support Vector Regression

    from sklearn.svm import SVR
    svr = make_pipeline(StandardScaler(), SVR(epsilon=0.01, kernel='poly'))
    save_regressor_and_metrics_report(svr,X,Y,"SVM Regresyon")

    ### XGBOOSTgünell
    from xgboost import XGBRegressor
    xgbr = XGBRegressor(colsample_bytree= 0.8, gamma= 0, max_depth= 5, min_child_weight= 1, n_estimators= 50, reg_alpha= 0.1, reg_lambda= 1)
    save_regressor_and_metrics_report(xgbr,X,Y,"XGBoost Regresyon")

def retrain_model(path,type,keep_best=True):
    df = engine.get_table(as_df=True)
    df["product_code"] = df["product_code"].astype(str)
    X = df.drop(["energy_value","energy_class"],axis=1)
    X = loader.load_preprocessor().transform(X)
    Y = df[["energy_value","energy_class"]]

    from sklearn.base import clone
    model = loader.load(path)
    cloned = clone(model["model"])

    if type == "classification":
        cloned,metrics = get_classification_metrics(cloned,X,Y["energy_class"])
        old_score = model["metrics"]["f1"]
        new_score = metrics["f1"]

    elif type == "regression":
        cloned,metrics = regressor_report(cloned,X,Y["energy_value"])
        old_score = model["metrics"]["r2"]
        new_score = metrics["r2"]


    if keep_best:
        if new_score >= old_score:
            model["model"] = cloned
            model["metrics"] = metrics
            loader.change(path, model)

    else:
        model["model"] = cloned
        model["metrics"] = metrics
        loader.change(path, model)

if __name__ == "__main__":
    train_classifiers()
    train_regressors()

    


    











