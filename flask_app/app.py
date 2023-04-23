from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np
import os
import lime
import dill
import matplotlib
import matplotlib.pyplot as plt

app = Flask(__name__)


current_dir = os.path.dirname(os.path.abspath(__file__))
input_file_path = os.path.join(current_dir, 'inputs', 'inputs_flask_app_dict.pkl')

with open(input_file_path, 'rb') as f:
    inputs = pickle.load(f)

with open(os.path.join(current_dir, 'inputs', 'explainer.pkl'), 'rb') as f:
    lime_explainer = dill.load(f)         


@app.route('/')
def home():  
    features = inputs['final_features']
    categorical_features = inputs['categorical_features']
    return render_template('index.html', variables = features,
                           categorical_features = categorical_features)


def lime_plot(input, model):

    def custom_lime_plot(exp, pos_color: str = 'red', neg_color: str = 'green'):

        fig, ax = plt.subplots()
        vals = [i[1] for i in exp.as_list()][::-1]
        names = [i[0].split(' <=')[0].split('< ')[-1].split(' >')[0] for i in exp.as_list()][::-1]

        colors = [pos_color if x > 0 else neg_color for x in vals]
        pos = np.arange(len(vals)) + .5

        ax.barh(pos, vals, align = 'center', color = colors)
        ax.set_yticks(pos)
        ax.set_yticklabels(names)

        return fig
    
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    
    exp = lime_explainer.explain_instance(
                                        data_row = input,
                                        predict_fn = model.predict_proba
                                        )

    fig = custom_lime_plot(exp)
    fig.set_size_inches(10, 6)

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('Feature', fontsize = 17)
    ax.set_xlabel('Contribution', fontsize = 17)
    plt.title('')

    fig.savefig(os.path.join(current_dir, 'static', 'lime_explanation.png'),
                format = 'png', bbox_inches = 'tight', dpi = 300, transparent = True)
    plt.close(fig)




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

    lime_plot(input_df_woe_FINAL.iloc[0], model)

    return render_template('results.html',
                           prediction = round(pred_score[0] * 100, 2),
                           predicted_class = result[0])


if __name__ == '__main__':
    app.run(debug = True)