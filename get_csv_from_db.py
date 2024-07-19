from engine import engine
import pandas as pd

data = engine.get_table(as_df=True)
data.to_csv("data.csv")