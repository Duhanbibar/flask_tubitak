from flask import Blueprint, render_template, request
import pandas as pd
from pickle_needs import *
from loader import get_preprocessor,get_models

# Create a Blueprint
main = Blueprint('module3', __name__)
def classification_report(product):
    _,classifiers = get_models()
    for model in classifiers["base"]:
        model["prediction"] = model["model"].predict(product)

    for model in classifiers["user"]:
        model["prediction"] = model["model"].predict(product)
    
    return render_template("classification_report.html",bases=classifiers["base"],users=classifiers["user"])



def regression_report(product):
    regressors,_ = get_models()
    for model in regressors["base"]:
        model["prediction"] = model["model"].predict(product)

    for model in regressors["user"]:
        model["prediction"] = model["model"].predict(product)
    
    return render_template("regression_report.html",bases=regressors["base"],users=regressors["user"])


@main.route('/tahmin', methods = ["POST","GET"])
def tahmin():
    
    if request.method == 'POST':
        product_data = request.form.to_dict(flat=False)
        product = pd.DataFrame(product_data)
        product = product.drop(["tahminci"],axis = 1)
        product = get_preprocessor().transform(product)

        tahminci = request.form.get("tahminci")

        if tahminci == "Regression":
            return regression_report(product)

        elif tahminci == "Classification":
            return classification_report(product)

        
    else:
        return render_template("tahmin.html", kontrol=None, error=None,data={})
    # If it's a GET request, handle it here
    # You can add GET request specific logic if needed
    return render_template("tahmin.html", error=None)



