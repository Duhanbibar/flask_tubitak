from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from joblib import load,dump
from engine import engine
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,precision_score,recall_score,f1_score
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error,r2_score
import numpy as np
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from loader import loader

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


def log(to_be_printed):

    with open(log_file,"a") as logs:
        sys.stdout = logs
        if type(to_be_printed) == pd.DataFrame:
            with pd.option_context("display.max_rows",None,
                                'display.max_columns',None,
                                'display.width',None,
                                'display.max_colwidth',None):
                print(to_be_printed)
        elif isinstance(to_be_printed, np.ndarray):
            with np.printoptions(threshold=np.inf, linewidth=np.inf):
                    print(to_be_printed)
        else:
            print(to_be_printed)

class LabelEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.label_encoder = LabelEncoder()
    
    def fit(self, y):
        self.label_encoder.fit(y)
        return self
    
    def transform(self, y):
        return self.label_encoder.transform(y)
    
    def inverse_transform(self, y):
        return self.label_encoder.inverse_transform(y)

class SVCWithLabelEncoding(BaseEstimator, ClassifierMixin):
    def __init__(self, classifier):
        self.classifier = classifier
        self.label_encoder = LabelEncoderTransformer()

    def fit(self, X, y):
        self.label_encoder.fit(y)
        y_encoded = self.label_encoder.transform(y)
        self.classifier.fit(X, y_encoded)
        return self
        
    def predict(self, X):
        y_encoded = self.classifier.predict(X)
        return self.label_encoder.inverse_transform(y_encoded)
    
    def predict_proba(self, X):
        return self.classifier.predict_proba(X)
    
class XGBCWithLabelEncoding(BaseEstimator, ClassifierMixin):
    def __init__(self, classifier):
        self.classifier = classifier
        self.label_encoder = LabelEncoderTransformer()

    def fit(self, X, y):
        self.label_encoder.fit(y)
        y_encoded = self.label_encoder.transform(y)
        self.classifier.fit(X, y_encoded)
        return self
        
    def predict(self, X):
        y_encoded = self.classifier.predict(X)
        return self.label_encoder.inverse_transform(y_encoded)
    
    def predict_proba(self, X):
        return self.classifier.predict_proba(X)
    

def save_regressor_and_metrics_report(model,X_test,y_test,model_name=None):
    global regressor_counter

    regression_eval = regressor_report(model,X_test,y_test)

    if not model_name:
        model_name = type(model).__name__

    data = {
        "model":model,
        "name": model_name,
        "metrics": regression_eval,
        "id": f"user_regressor{regressor_counter +1}"
    }
    regressor_counter +=1
    dump(data,f"regressors/user/{data['name']}.joblib")

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

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns)

def save_classifier_and_metrics_report(model,X_test,y_test,model_name=None):
    global classifier_counter
    classification_metrics = get_classification_metrics(model,X_test,y_test)

    if not model_name:
        model_name = type(model).__name__

    data = {
        "model":model,
        "name": model_name,
        "metrics": classification_metrics,
        "id": f"user_classifier{classifier_counter +1}"
    }
    classifier_counter += 1

    dump(data,f"classifiers/user/{data['name']}.joblib")

if __name__ == "__main__":
    with open(log_file,"w") as logs:
        logs

    result = engine.get_table()
    rows = [row for row in result]
    df = pd.DataFrame(rows)
    df["product_code"] = df["product_code"].astype(str)

    

    X = df.drop(["energy_value","energy_class"],axis=1)
    log(X)
    X = loader.load_preprocessor().transform(X)
    Y = df[["energy_value","energy_class"]]


    if not cv:
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

        ### Random Forests Regression
        from sklearn.ensemble import RandomForestRegressor
        rfr = make_pipeline(StandardScaler(),  RandomForestRegressor(max_depth= 10, max_features= "sqrt", min_samples_leaf= 2, min_samples_split= 2, n_estimators= 100))
        rfr.fit(X_train,y_train_values)
        save_regressor_and_metrics_report(rfr,X_test,y_test_values)

        ### Linear Regression
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        lr.fit(X_train,y_train_values)
        save_regressor_and_metrics_report(lr,X_test,y_test_values)

        ### Support Vector Regression

        from sklearn.svm import SVR
        svr = make_pipeline(StandardScaler(), SVR(epsilon=0.01, kernel='linear'))
        svr.fit(X_train,y_train_values)
        save_regressor_and_metrics_report(svr,X_test,y_test_values,"SVR")


        ### Support Vector Classification

        from sklearn.svm import SVC
        svc_with_label_encoding = SVCWithLabelEncoding(SVC())
        svc_pipeline = Pipeline(steps=[
            ('svc_with_label_encoding', svc_with_label_encoding)
        ])
        svc_pipeline.fit(X_train, y_train_labels)
        predictions = svc_pipeline.predict(X_test)
        log(predictions)
        save_classifier_and_metrics_report(svc_pipeline,X_test,y_test_labels,"SVC")

        ### Logiistic Regression
        from sklearn.linear_model import LogisticRegression
        logi = LogisticRegression()
        logi.fit(X_train,y_train_labels)
        save_classifier_and_metrics_report(logi,X_test,y_test_labels)


        ### XGBOOST
        from xgboost import XGBClassifier,XGBRegressor
        xgbr = XGBRegressor(max_depth=3,
                            gamma=0.0,
                            reg_alpha=0.1,
                            reg_lambda=10,
                            colsample_bytree=1,
                            min_child_weight=1,
                            n_estimators=100)
        xgbr.fit(X_train,y_train_values)
        save_regressor_and_metrics_report(xgbr,X_test,y_test_values)

        xgbc = XGBClassifier()
        xgbc_with_label_encoding = XGBCWithLabelEncoding(xgbc)
        xgbc_pipeline = Pipeline(steps=[
            ('xgbc_with_label_encoding', xgbc_with_label_encoding)
        ])
        xgbc_pipeline.fit(X_train, y_train_labels)
        save_classifier_and_metrics_report(xgbc_pipeline,X_test,y_test_labels,"XGBClassifier")



        

        ### More models should be implemented here.











