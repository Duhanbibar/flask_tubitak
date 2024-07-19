from joblib import load,dump
import os
from functools import wraps
from pickle_needs import *


def singleton(cls):
    instances = {}
    
    @wraps(cls)
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance

@singleton
class Loader:
    def __init__(self):
        self.loaded_files = {}

    def load(self, path):
        # Get the absolute path of the file
        absolute_path = os.path.abspath(path)
        
        if absolute_path in self.loaded_files:
            return self.loaded_files[absolute_path]
        else:
            # Load the file (replace with actual loading logic)
            self.loaded_files[absolute_path] = load(absolute_path)
        
        return self.loaded_files[absolute_path]
    
    def change(self,path,object):
        absolute_path = os.path.abspath(path)
        self.loaded_files[absolute_path] = object
        dump(object,absolute_path)

    def clear_cache(self):
        self.loaded_files = {}
        print("Cache cleared")
    
    def load_preprocessor(self):
        return self.load(os.path.join("preprocessors","preprocessor.joblib"))


loader = Loader()

def get_models():
    regressors = {
        "base":[],
        "user":[]
    }

    base_dir = os.path.join("regressors","base")
    user_dir = os.path.join("regressors","user")

    for filename in os.listdir(base_dir):
        regressors["base"].append(loader.load(os.path.join(base_dir,filename)))
    
    for filename in os.listdir(user_dir):
        regressors["user"].append(loader.load(os.path.join(user_dir,filename)))

    base_dir = os.path.join("classifiers","base")
    user_dir = os.path.join("classifiers","user")
    classifiers = {
        "base":[],
        "user":[]
    }
    for filename in os.listdir(base_dir):
        classifiers["base"].append(loader.load(os.path.join(base_dir,filename)))
    
    for filename in os.listdir(user_dir):
        classifiers["user"].append(loader.load(os.path.join(user_dir,filename)))

    return regressors,classifiers

def get_preprocessor():
    return loader.load(os.path.join("preprocessors","preprocessor.joblib"))
