from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from engine import get_table
import joblib
import pandas as pd
import sys
import numpy as np
from loader import Loader
from logger import Logger

"""
The preprocessor pipeline that will transform the dataframe into desired training data, details at: 
https://github.com/1942Spectre/Tubitak_Energytest/blob/main/Data%20Pipeline.ipynb


Expected Dataframe Structure can be seen at expected_dataframe_structure.txt:
"""

logger = Logger("logs/preprocessor.log")

cooling_fan_categorical_columns = ["cooling_fan_type"]
cooling_fan_numerical_columns = ["cooling_fan_power","cooling_fan_rpm"]


cooling_fan_categorical_transformer = Pipeline(
    steps = [
    ("imputer", SimpleImputer(strategy="constant",fill_value = "No Cooling Fan")), ## Impute Null Values with "No Cooling Fan"
    ("onehot",OneHotEncoder(handle_unknown = "ignore")) # Encode Categorical Values
    ])

cooling_fan_numeric_transformer = Pipeline(
    steps = [
    ("imputer", SimpleImputer(strategy = "constant", fill_value = 0)), ## Impute Null Values with 0
    #("scaler", StandardScaler()) ## Scale Numeric Features 
    ])

ventilation_channel_pipeline = Pipeline(
    steps = [
        ("imputer", SimpleImputer(strategy="constant",fill_value = "No Ventilation Channel")),
        ("One hot encode",OneHotEncoder(handle_unknown="ignore"))
])

columns_to_be_dropped = ["product_code"]

class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns)
    

columns_to_be_encoded = ['test_function','pulsation','buffle','cavity_type','bracket_description']

onehot_encoder_transformer = Pipeline(
    steps = [
        ("onehot",OneHotEncoder(handle_unknown = "ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers = [
        ("Drop Product Code" , DropColumns(columns = columns_to_be_dropped), columns_to_be_dropped),
        ("Ventilation Channel", ventilation_channel_pipeline, ['ventilation_channel']),
        ("Cooling Fan Type",cooling_fan_categorical_transformer,cooling_fan_categorical_columns),
        ("Cooling Fan Power and Cooling Fan RPM",cooling_fan_numeric_transformer,cooling_fan_numerical_columns),
        ("Other Categorical Columns To Be Encoded",onehot_encoder_transformer,columns_to_be_encoded)
    ])


db = get_table(as_df=True)
X = db.drop(["energy_value","energy_class"],axis =1)

a = preprocessor.fit_transform(X)
joblib.dump(preprocessor,"preprocessors/preprocessor.joblib")

loader = Loader()
def load_preprocessor():
    return loader.load("preprocessors/preprocessor.joblib")