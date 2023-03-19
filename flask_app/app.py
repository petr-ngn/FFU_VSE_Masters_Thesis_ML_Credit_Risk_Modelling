from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
import os


app = Flask(__name__)


current_dir = os.path.dirname(os.path.abspath(__file__))
input_file_path = os.path.join(current_dir, 'inputs', 'inputs_flask_app_dict.pkl')

with open(input_file_path, 'rb') as f:
    inputs = pickle.load(f)
            

@app.route('/')
def home():  
    features = inputs['final_features']
    categorical_features = inputs['categorical_features']
    return render_template('index.html', variables = features, categorical_features = categorical_features)


@app.route('/predict', methods = ['POST'])
def predict():
    woe_bins, woe_binning, model, threshold, final_features, categorical_features, input_df = (input for _, input in inputs.items())

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

    input_df_woe = woe_binning.transform(input_df, metric = 'woe')

    for feature in input_df_woe.columns:
        na_woe = woe_bins.query('Variable == @feature and Bin == "Missing"')['WoE'].values[0]
        input_df_woe.loc[input_df[feature].isna(), feature] = na_woe

    input_df_woe_FINAL = input_df_woe[final_features]
    pred_score = model.predict_proba(input_df_woe_FINAL)[:, 1]

    result = ['Loan application denied' if i > threshold else 'Loan application approved' for i in pred_score]

    return render_template('results.html',
                           prediction = round(pred_score[0] * 100, 2),
                           predicted_class = result[0])

if __name__ == '__main__':
    app.run(debug = True)
