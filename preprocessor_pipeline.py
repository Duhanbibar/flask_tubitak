from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from engine import get_table
import joblib
from loader import Loader
from logger import Logger
from pickle_needs import *
from sklearn.decomposition import PCA
from sklearn.compose import make_column_selector





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
columns_to_be_encoded = ['test_function','pulsation','buffle','cavity_type','bracket_description','function_type']

onehot_encoder_transformer = Pipeline(
    steps = [
        ("onehot",OneHotEncoder(handle_unknown = "ignore"))
    ]
)
## Cooling Fan PCA
pca_cooling_fan = Pipeline(
    steps = [
        ("pca_cooling_fan",PCA(n_components=1))
    ]
)

## Insulation density/thickness PCA
pca_insulation = Pipeline(
    steps = [
        ("pca_insulation",PCA(n_components=1))
    ]
)

numeric_scaler = Pipeline(
    steps=[
        ("scaler",StandardScaler())
    ]
)

from sklearn.decomposition import PCA

preprocessor = ColumnTransformer(
    transformers = [
        ("Drop Product Code" , DropColumns(columns = columns_to_be_dropped), columns_to_be_dropped),
        ("Ventilation Channel", ventilation_channel_pipeline, ['ventilation_channel']),
        ("Cooling Fan Type",cooling_fan_categorical_transformer,cooling_fan_categorical_columns),
        ("Cooling Fan Power and Cooling Fan RPM",cooling_fan_numeric_transformer,cooling_fan_numerical_columns),
        ("Other Categorical Columns To Be Encoded",onehot_encoder_transformer,columns_to_be_encoded),
        #("scaler", numeric_scaler, make_column_selector(dtype_include=['number'])),  # Apply to all numerical columns
        #("pca_cooling_fan", pca_cooling_fan,[*cooling_fan_numerical_columns]), ### Pca only transforms numerics for now.
        #("pca_insulation", pca_insulation,["insulation_density","insulation_thickness"]) ### Pca only transforms numerics for now.
     ],remainder="passthrough")



db = get_table(as_df=True)
X = db.drop(["energy_value","energy_class"],axis =1)
a = preprocessor.fit_transform(X)


joblib.dump(preprocessor,"preprocessors/preprocessor.joblib")

loader = Loader()
def load_preprocessor():
    return loader.load("preprocessors/preprocessor.joblib")