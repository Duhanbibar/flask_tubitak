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

    regression_eval = regressor_report(model,X_test,y_test)

    if not model_name:
        model_name = type(model).__name__

    data = {
        "model":model,
        "name": model_name,
        "metrics": regression_eval,
        "id": f"{save_model_owner}_regressor{regressor_counter +1}",
        "path":f"regressors/{save_model_owner}/{model_name}.joblib"
    }
    print(f"{model_name}{data['metrics']['r2']}")
    regressor_counter +=1
    dump(data,f"regressors/{save_model_owner}/{model_name}.joblib")


def save_classifier_and_metrics_report(model,X_test,y_test,model_name=None):
    global classifier_counter,save_model_owner
    classification_metrics = get_classification_metrics(model,X_test,y_test)

    if not model_name:
        model_name = type(model).__name__

    data = {
        "model":model,
        "name": model_name,
        "metrics": classification_metrics,
        "id": f"{save_model_owner}_classifier{classifier_counter +1}",
        "path":f"classifiers/{save_model_owner}/{model_name}.joblib"
    }
    print(f"{data['name']}{data['metrics']['f1']}")
    classifier_counter += 1

    dump(data,f"classifiers/{save_model_owner}/{model_name}.joblib")




def get_classification_metrics(model,X_test,y_test):
    y_pred = model.predict(X_test)
    return({
        "accuracy" : accuracy_score(y_test,y_pred),
        "confusion_mat" : confusion_matrix(y_test,y_pred),
        "precision" : precision_score(y_test,y_pred,average="binary", pos_label="A+"),
        "recall" : recall_score(y_test,y_pred,average="binary", pos_label="A+"),
        "class_report" : classification_report(y_test,y_pred),
        "f1" : f1_score(y_test,y_pred,average="binary", pos_label="A+")
    })

def regressor_report(model,X_test,y_test):
    y_pred = model.predict(X_test)
    return({
        "mse": mean_squared_error(y_test,y_pred),
        
        "rmse": np.sqrt(mean_squared_error(y_test,y_pred)),

        "mae": mean_absolute_error(y_test,y_pred),
        
        "mape": mean_absolute_percentage_error(y_test,y_pred),

        "r2": r2_score(y_test,y_pred),
    })

df = engine.get_table(as_df=True)
df["product_code"] = df["product_code"].astype(str)
X = df.drop(["energy_value","energy_class"],axis=1)
X = loader.load_preprocessor().transform(X)
Y = df[["energy_value","energy_class"]]

X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2)
y_train_values = y_train["energy_value"]
y_test_values = y_test["energy_value"]
y_train_labels = y_train["energy_class"]
y_test_labels = y_test["energy_class"]




def train_classifiers():
    ### Random Forests Classification
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier()
    rfc.fit(X_train,y_train_labels)
    save_classifier_and_metrics_report(rfc,X_test,y_test_labels)
    ### Support Vector Classification

    from sklearn.svm import SVC
    svc_with_label_encoding = SVCWithLabelEncoding(SVC())
    svc_pipeline = Pipeline(steps=[
        ('svc_with_label_encoding', svc_with_label_encoding)
    ])
    svc_pipeline.fit(X_train, y_train_labels)
    save_classifier_and_metrics_report(svc_pipeline,X_test,y_test_labels,"SVC")
    ### Logiistic Regression
    from sklearn.linear_model import LogisticRegression
    logi = LogisticRegression(max_iter = 100000)
    logi.fit(X_train,y_train_labels)
    save_classifier_and_metrics_report(logi,X_test,y_test_labels)

    from xgboost import XGBClassifier
    xgbc = XGBClassifier()
    xgbc_with_label_encoding = XGBCWithLabelEncoding(xgbc)
    xgbc_pipeline = Pipeline(steps=[
        ('xgbc_with_label_encoding', xgbc_with_label_encoding)
    ])
    xgbc_pipeline.fit(X_train, y_train_labels)
    save_classifier_and_metrics_report(xgbc_pipeline,X_test,y_test_labels,"XGBClassifier")

def train_regressors():
    ### Random Forests Regression
    from sklearn.ensemble import RandomForestRegressor
    rfr = RandomForestRegressor(max_depth= 10, max_features= "sqrt", min_samples_leaf= 2, min_samples_split= 2, n_estimators= 100)
    rfr.fit(X_train,y_train_values)
    save_regressor_and_metrics_report(rfr,X_train,y_train_values,"RandomForestRegression")

    ### Linear Regression
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X_train,y_train_values)
    save_regressor_and_metrics_report(lr,X_train,y_train_values)

    ### Support Vector Regression

    from sklearn.svm import SVR
    svr = make_pipeline(StandardScaler(), SVR(epsilon=0.01, kernel='poly'))
    svr.fit(X_train,y_train_values)
    save_regressor_and_metrics_report(svr,X_train,y_train_values,"SVR")

    ### XGBOOST
    from xgboost import XGBClassifier,XGBRegressor
    xgbr = XGBRegressor(colsample_bytree= 0.8, gamma= 0, max_depth= 5, min_child_weight= 1, n_estimators= 50, reg_alpha= 0.1, reg_lambda= 1)
    xgbr.fit(X_train,y_train_values)
    save_regressor_and_metrics_report(xgbr,X_train,y_train_values)

def retrain_model(path,type):
    from sklearn.base import clone
    print("\n\n\n")
    print(path)
    model = loader.load(path)
    print(model)
    cloned = clone(model["model"])

    if type == "classification":
        old_score = model["metrics"]["f1"]
        cloned.fit(X_train,y_train_labels)
        y_pred = cloned.predict(X_test)
        score = f1_score(y_test_labels,y_pred,average="binary", pos_label="A+")
        model["model"] = cloned
        model["metrics"] = get_classification_metrics(cloned,X_test,y_test_labels)

    elif type == "regression":
        old_score = model["metrics"]["r2"]
        cloned.fit(X_train,y_train_values)
        y_pred = cloned.predict(X_train)
        score = r2_score(y_train_values,y_pred)
        model["model"] = cloned
        model["metrics"] = regressor_report(cloned,X_train,y_train_values)

    if score >= old_score:
        loader.change(path, model)
    
    


    











