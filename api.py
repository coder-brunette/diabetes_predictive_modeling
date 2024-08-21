from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the trained model
model = joblib.load('diabetes_model.pkl')

# Load the scaler used during training
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json(force=True)

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Scale the input data
        df_scaled = scaler.transform(df)

        # Make predictions
        prediction = model.predict(df_scaled)
        probability = model.predict_proba(df_scaled)

        # Prepare response
        response = {
            'prediction': int(prediction[0]),
            'probability': probability[0].tolist()
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)


#  curl -X POST http://127.0.0.1:5000/predict \
#     -H "Content-Type: application/json" \
#     -d '{"Pregnancies": 2, "Glucose": 120, "BloodPressure": 70, "SkinThickness": 30, "Insulin": 0, "BMI": 32.0, "DiabetesPedigreeFunction": 0.5, "Age": 45}'


'''
{
  "prediction": 1,
  "probability": [0.48,0.52]
}
'''