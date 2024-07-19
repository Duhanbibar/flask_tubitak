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
X = preprocessor.transform(X)


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)

y_train_labels = Y_train["energy_class"]
y_test_labels= Y_test["energy_class"]

y_train_values= Y_train["energy_value"]
y_test_values= Y_test["energy_value"]


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train_labels)

save_classifier_and_metrics_report(knn,X_test,y_test_labels,model_name="knn_example.joblib")

from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor(n_neighbors=5)
knr.fit(X_train,y_train_values)

save_regressor_and_metrics_report(knr,X_test,y_test_labels,model_name="knr_example.joblib")
