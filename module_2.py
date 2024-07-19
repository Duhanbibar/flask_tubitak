from flask import Blueprint, render_template,request,redirect,url_for
from logger import Logger
from loader import get_models


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
    model = request.data
    logger.log(model)
    return redirect(url_for("module2.retrain_models"))

