from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
import os


app = Flask(__name__)

@app.route('/')
def home():
    # Get the absolute path of the directory where this Python script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the input file relative to the current directory
    input_file_path = os.path.join(current_dir, 'inputs', 'inputs_flask_app_dict.pkl')
    with open(input_file_path, 'rb') as f:
         inputs = pickle.load(f)
    features = inputs['final_features']

    return render_template('index.html', variables = features, categorical_features = ['JOB','REASON'])


@app.route('/predict', methods=['POST'])
def predict():
    current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the absolute path to the input file relative to the current directory
    input_file_path = os.path.join(current_dir, 'inputs', 'inputs_flask_app_dict.pkl')
    with open(input_file_path, 'rb') as ff:
        inputs = pickle.load(ff)

    woe_bins, woe_binning, model, threshold, final_features, input_df = (input for _, input in inputs.items())

    for feature in input_df.columns:
        if feature in final_features:
            if feature in ['REASON', 'JOB']:
                input_df.loc[0, feature] = str(request.form[feature]) if request.form[feature] else np.nan
            elif feature == 'LOAN':
                input_df.loc[0, feature] = int(request.form[feature]) if request.form[feature] else np.nan
            else:
                input_df.loc[0, feature] =  float(request.form[feature]) if request.form[feature] else np.nan
        else:
            input_df.loc[0, feature] = np.nan

    
    data_ = woe_binning.transform(input_df, metric = 'woe')

    for feature in data_.columns:
        na_woe = woe_bins.query('Variable == @feature and Bin == "Missing"')['WoE'].values[0]
        data_.loc[input_df[feature].isna(), feature] = na_woe

    data_ = data_[final_features]
    prediction = model.predict_proba(data_)[:, 1]

    result = ['Loan application denied' if i > threshold else 'Loan application approved' for i in prediction]
    # Render the results page with the predicted probability
    return render_template('results.html', prediction=round(prediction[0]*100,2), predicted_class = result[0])

if __name__ == '__main__':
    app.run(debug = True)
