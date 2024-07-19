import joblib
import os

if __name__ == "__main__":
    base_dir = os.path.join("regressors","base")
    user_dir = os.path.join("regressors","user")

    for filename in os.listdir(base_dir):
        x = joblib.load(os.path.join(base_dir,filename))
        x["path"] = os.path.join(base_dir,filename)
        joblib.dump(x,os.path.join(base_dir,filename))
    
    for filename in os.listdir(user_dir):
        x = joblib.load(os.path.join(user_dir,filename))
        x["path"] = os.path.join(user_dir,filename)
        joblib.dump(x,os.path.join(user_dir,filename))