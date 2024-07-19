from flask import Blueprint, render_template,request,redirect,url_for
from logger import Logger
from pickle_needs import *
from loader import get_models
from model_builder import train_regressors,train_classifiers
from model_builder import retrain_model as retrainer


# Create a Blueprint
main = Blueprint('module2', __name__)
logger = Logger()



@main.route("/retrain",methods = ['GET'])
def retrain_models():
    regressors,classifiers = get_models()
    bases = {"regression":regressors["base"],
             "classification":classifiers["base"]}
    
    users = {"regression":regressors["user"],
             "classification":classifiers["user"]}
    return render_template("retrain.html",base_models = bases,user_models = users)


@main.route("/retrain_model",methods = ["POST"])
def retrain_model():
    path = request.form.get("path")
    type = request.form.get("type")
    retrainer(path,type)
    return redirect(url_for("module2.retrain_models"))

@main.route("/retrain_classifiers",methods = ["GET"])
def retrain_classifiers():
    train_classifiers()
    return redirect(url_for("module2.retrain_models"))

@main.route("/retrain_regressors",methods = ["GET"])
def retrain_regressors():
    train_regressors()
    return redirect(url_for("module2.retrain_models"))


