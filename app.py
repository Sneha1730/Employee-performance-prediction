from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
scaler = joblib.load("scaler.pkl")  # typo corrected to "scaler.pkl" if that's the real filename
model = joblib.load("model_performance.pkl")

@app.route("/")
def home():
    return render_template("form.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get values from form
        years = float(request.form["Years_At_Company"])
        salary = float(request.form["Monthly_Salary"])
        overtime = float(request.form["Overtime_Hours"])
        promotion = int(request.form["Promotions"])
        satisfaction = float(request.form["Employee_Satisfaction_Score"])

        # Format the input
        X = np.array([[years, salary, overtime, promotion, satisfaction]])
        X_scaled = scaler.transform(X)

        # Make prediction
        prediction = model.predict(X_scaled)[0]

        return render_template("form.html", prediction=prediction)
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
