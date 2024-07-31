from loader import *
from engine import *
from logger import *

models = get_models()
model = (models[0]["user"][-1])["model"]

df = engine.get_table(as_df=True)
df["product_code"] = df["product_code"].astype(str)
X = df.drop(["energy_value","energy_class"],axis=1)
X = loader.load_preprocessor().transform(X)
Y = df[["energy_value","energy_class"]]

df["prediction"] = model.predict(X)
df.to_csv("prediction_result.csv",sep=";")

logger = Logger()
logger.log(df)