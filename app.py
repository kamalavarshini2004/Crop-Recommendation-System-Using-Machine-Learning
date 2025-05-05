import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from flask import Flask, request, render_template
import numpy as np
import pickle
import os

app = Flask(__name__)

# Load model and scalers safely
try:
    model = pickle.load(open('model.pkl', 'rb'))
    sc = pickle.load(open('standscaler.pkl', 'rb'))
    ms = pickle.load(open('minmaxscaler.pkl', 'rb'))
except Exception as e:
    raise RuntimeError(f"Failed to load model files: {str(e)}")

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    try:
        # Validate and convert inputs
        required_params = ['Nitrogen', 'Phosporus', 'Potassium', 
                          'Temperature', 'Humidity', 'Ph', 'Rainfall']
        feature_list = []
        
        for param in required_params:
            value = request.form.get(param)
            if value is None or value.strip() == '':
                return render_template("index.html", 
                                     result=f"Missing or invalid value for {param}")
            try:
                feature_list.append(float(value))
            except ValueError:
                return render_template("index.html", 
                                     result=f"Invalid number format for {param}")

        # Prepare features
        single_pred = np.array(feature_list).reshape(1, -1)

        # Apply scalers
        scaled_features = ms.transform(single_pred)
        final_features = sc.transform(scaled_features)

        # Predict
        prediction = model.predict(final_features)

        # Label mapping
        crop_dict = {
            1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
            6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon",
            10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
            14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
            17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
            20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
        }

        crop = crop_dict.get(int(prediction[0]), "Unknown")
        result = f"{crop} is the best crop to be cultivated right there"

    except Exception as e:
        result = f"Prediction error: {str(e)}"
        app.logger.error(f"Prediction failed: {str(e)}")

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False)  # debug=False for production