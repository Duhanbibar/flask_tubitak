from sklearn.metrics import mean_absolute_error,mean_squared_error,f1_score,r2_score,accuracy_score,precision_score,recall_score,confusion_matrix,classification_report
import numpy as np
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.base import clone
from logger import Logger
logger = Logger("logs/kfold_logs")
def regressor_report(model, X, y):
    # Initialize GroupKFold with 10 splits
    kf = KFold(n_splits=10,shuffle=True)
    model = clone(model)
    
    # Initialize lists to store metrics for each fold
    mses = []
    rmses = []
    maes = []
    mapes = []
    r2s = []

    # Split the data into training and testing sets for each fold
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train the model
        model.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = model.predict(X_test)
        
        # Calculate and store metrics
        mse = mean_squared_error(y_test, y_pred)
        mses.append(mse)
        rmses.append(np.sqrt(mse))
        maes.append(mean_absolute_error(y_test, y_pred))
        mapes.append(np.mean(np.abs((y_test - y_pred) / y_test)) * 100)
        r2s.append(r2_score(y_test, y_pred))
    logger.log(r2s)
    print(r2s)

    # Return the average of the metrics over all folds
    return model,{
        "mse": np.mean(mses),
        "rmse": np.mean(rmses),
        "mae": np.mean(maes),
        "mape": np.mean(mapes),
        "r2": np.mean(r2s)
    }



def get_classification_metrics(model, X, y):
    model = clone(model)
    # Initialize StratifiedGroupKFold with 10 splits
    skf = StratifiedKFold(n_splits=10,shuffle=True)
    
    # Initialize lists to store metrics for each fold
    accuracies = []
    precisions = []
    recalls = []
    f1s = []

    # Split the data into training and testing sets for each fold
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train the model
        model.fit(X_train, y_train)
        
        # Predict on the test set
        y_pred = model.predict(X_test)
        
        # Calculate and store metrics
        accuracies.append(accuracy_score(y_test, y_pred))
        precisions.append(precision_score(y_test, y_pred, average="binary", pos_label="A+"))
        recalls.append(recall_score(y_test, y_pred, average="binary", pos_label="A+"))
        
        f1s.append(f1_score(y_test, y_pred, average="binary", pos_label="A+"))

    # Return the average of the metrics over all folds
    return model,{
        "accuracy": np.mean(accuracies),
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "f1": np.mean(f1s)
    }
