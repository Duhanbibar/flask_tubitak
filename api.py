from flask import Blueprint, request,jsonify
from engine import get_product_for_prediction


# Create a Blueprint
main = Blueprint('api', __name__,url_prefix="")

@main.route("/get_product_data",methods = ["POST"])
def get_product_data():
    print("hi")
    data = request.get_json()
    product_code = str(data.get("product_code"))
    product_data = get_product_for_prediction(product_code)

    if not product_data.empty:
        json_data = product_data.to_dict(orient = "records")[0]
        return jsonify({'success': True, 'data': json_data})

    else:
        response = {
            'success':False
        }
    
    return jsonify(response)