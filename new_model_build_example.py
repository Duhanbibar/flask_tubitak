from model_builder import save_classifier_and_metrics_report,save_regressor_and_metrics_report
from loader import loader
from engine import get_table
from pickle_needs import *

preprocessor = loader.load_preprocessor()
#### End Of parts

### Tabloyu getir
df = get_table(as_df=True)

### X ve y ayrımı

X = df.drop(["energy_value","energy_class"],axis=1)
Y = df[["energy_class","energy_value"]]
y_labels = df["energy_class"]
y_values = df["energy_value"]
X = preprocessor.transform(X)


from sklearn.model_selection import train_test_split



from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

save_classifier_and_metrics_report(knn,X,y_labels,model_name="knn_example.joblib")

from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor(n_neighbors=5)

save_regressor_and_metrics_report(knr,X,y_values,model_name="knr_example.joblib")



