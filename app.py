from flask import Flask
from joblib import load
from pickle_needs import *
from module_1 import main as module_1
from module_2 import main as module_2
from module_3 import main as module_3
from api import main as api
from main import main 
app = Flask(__name__)
app.register_blueprint(module_1)
app.register_blueprint(module_2)
app.register_blueprint(module_3)
app.register_blueprint(api)
app.register_blueprint(main)


if __name__ == '__main__':
    app.run(debug=True)




    

