from flask import Blueprint, render_template

# Create a Blueprint
main = Blueprint('main', __name__,url_prefix="")



@main.route('/', methods=['GET'])
def index():
    return render_template('index.html')
