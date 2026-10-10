
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("house_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            area = float(request.form["area"])
            bedrooms = int(request.form["bedrooms"])
            bathrooms = int(request.form["bathrooms"])
            age = float(request.form["age"])
            location = request.form["location"]

            if area <= 0:
                raise ValueError

            if bedrooms <= 0 or bathrooms <= 0:
                raise ValueError

            if age < 0:
                raise ValueError

            if location not in ["City", "Suburb"]:
                raise ValueError

            house = pd.DataFrame([{
                "area": area,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "age": age,
                "location": location
            }])

            predicted_price = model.predict(house)[0]
            prediction = f"₹{predicted_price:,.0f}"

        except (ValueError, TypeError):
            error = "Please enter valid house details."

    return render_template(
        "index.html",
        prediction=prediction,
        error=error
    )

if __name__ == "__main__":
    app.run(debug=True)
