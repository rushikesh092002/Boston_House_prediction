import json
import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

xgbmodel = pickle.load(open('XGB.pkl', 'rb'))
scaler = pickle.load(open('sc.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']
        print("Received data:", data)

        new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))

        output = xgbmodel.predict(new_data)
        print("Prediction:", output[0])

        return jsonify({'prediction': float(output[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict' ,methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input= scaler.transform(np.array(data).reshape(1, -1))
    print(final_input) 
    output = xgbmodel.predict(final_input)[0]
    return render_template('home.html', prediction="The House price prediction is {}".format(output))
        

if __name__ == '__main__':
    app.run(debug=True)

    
