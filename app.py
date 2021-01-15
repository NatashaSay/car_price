from flask import Flask, render_template, request, url_for, jsonify
from script import pred
import numpy as np
import pandas as pd


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST", "GET"])
def predict():
    if request.method == "POST":
        features = [x for x in request.form.values()]
        data = {"Id": 1, "engine_capacity": features[0], "type": features[1],
                "registration_year": features[2], "gearbox": features[3], "power":features[4],
                "model": features[5], "mileage": features[6], "fuel": features[7],
                "brand": features[8], "damage": features[9], "insurance_price": features[10]}

        df = pd.DataFrame(data, index=[0])
        df.to_csv("./tables/test_no_target1.csv", sep=',',index=False)
        price = pred()
        price = round(price[0],2)
        return render_template("index1.html", mark = price, data=data)

    return render_template("index.html")



@app.route("/version")
def version():
    return jsonify({'model': 'v1-2021-01-15_18:26:53.386362.model', 'version': '1.0'})


if __name__ == '__main__':
    app.run(debug=True)
