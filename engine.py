import sqlalchemy
from sqlalchemy import text
import pandas as pd
from loader import singleton


database = True

if database:
    @singleton
    class Connection():
        def __init__(self):
            self.engine = sqlalchemy.create_engine(
                "mssql+pyodbc://DESKTOP-3P2PGTE\\SQLEXPRESS/RENTA1505?"
                "driver=ODBC+Driver+18+for+SQL+Server&TrustServerCertificate=yes"
                "&authentication=ActiveDirectoryIntegrated",echo=True
            )

        def get_table(self,as_df = False):
            with self.engine.connect() as conn:
                query= text("SELECT * from dbo.data ")
                result = conn.execute(query).fetchall()
                if not as_df:
                    return result
                rows = [row for row in result]
            return pd.DataFrame(rows)

        def get_product(self,model_no):
            with self.engine.connect() as conn:
                query= text(f"SELECT * from dbo.data where model_no = {model_no} ")
                result = conn.execute(query).fetchall()
                rows = [row for row in result]
            return pd.DataFrame(rows).head(1)

        def save_data(self,data):
            print(data)
            data.to_sql("data",self.engine,if_exists="append",index=False)

    engine = Connection()

else:
    class Connection():
        def __init__(self):
            pass

        def get_table(self,as_df = False):
            df = pd.read_csv("data.csv").drop(["Unnamed: 0"],axis=1)
            if as_df:
                return df
            rows = [row for row in df]
            return rows

        def save_data(self,data):
            pass

    engine = Connection()


print(engine.get_table())
def get_product_for_prediction(product_code):
    df = engine.get_table(as_df = True)
    return df[df["product_code"] == product_code].drop(['energy_value','energy_class'],axis = 1).head(1)

def get_table(*args,**kwargs):
    return engine.get_table(*args,**kwargs)