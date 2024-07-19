from flask import Blueprint, render_template, request
from engine import engine,get_product_for_prediction


# Create a Blueprint
main = Blueprint('module1', __name__)

@main.route('/new_test', methods=['GET',"POST"])
def new_test():
        if request.method == "GET":
            return render_template("new_test.html")
        
        else:
            product_code = request.form.get("product_code")
            test_method = request.form.get("test_method")
            value = request.form.get("value")
            label = request.form.get("label")
            product = get_product_for_test_save(product_code)

            if not product.empty:
                
                product[test_method] = test_method
                product["energy_value"] = value
                product["energy_class"] = label
                engine.save(product.to_json())
                successes = ["New Test Saved Succesfully"]
                return render_template("new_test.html", successes = successes)


            else:
                errors = ["No product with that code is present at the database."]
                return render_template("new_test.html",errors = errors)
            #save_product(product)


@main.route('/save_product', methods=['POST'])
def save_product():
    error = ''
    product_code = request.form.get('product_code')
    energy_consumption = request.form.get('energy_consumption')
    Test_Function= request.form.get('test_function')
    if not product_code:
        error='Lütfen Ürün Kodunu Giriniz'
    if not energy_consumption:
        error='Lütfen Enerji Tüketimini girin'
    if not Test_Function:
        error='Lütfen Test Fonksiyonunu Seçin'
    if not get_product_for_prediction(product_code):
        error='Ürün Kodunuz Sistemde Bulunamamıştır.'
    if error:
        return render_template('new_product.html',error=error)
    
    # x=test_kaydet(product_code,energy_consumption,Test_Function)
    kontrol='Veri Kaydedilmiştir'
    return render_template('new_product.html' ,kontrol=kontrol,error=None)




def get_product_for_test_save(product_code):
    df = engine.get_table(as_df = True)
    return df[df["product_code"] == product_code].head(1)