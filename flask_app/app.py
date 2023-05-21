import warnings
warnings.filterwarnings("ignore")
from flask import Flask, render_template, request
import pandas as pd
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
    inputs = dill.load(f)

lime_explainer = inputs["lime_explainer"]      


def lime_plot(input, model, lime_explainer):

    def custom_lime_plot(exp, pos_color: str = 'red', neg_color: str = 'green'):

        fig, ax = plt.subplots()
        vals = [i[1] for i in exp.as_list()][::-1]
        names = [i[0].split(' <=')[0].split('< ')[-1].split(' >')[0] for i in exp.as_list()][::-1]
        feat_dict = {'JOB': 'Job occupancy',
                    'REASON': 'Reason of loan application',
                    'LOAN': 'Requested loan amount',
                    'MORTDUE': 'Amount due on existing mortgage',
                    'VALUE': 'Current property value',
                    'YOJ': 'Years at present job',
                    'DEROG': '# of major derogatory reports',
                    'DELINQ': '# of delinquent credit lines',
                    'CLAGE': 'Age of the oldest credit line',
                    'NINQ': '# of recent credit inquiries',
                    'CLNO': '# of credit lines',
                    'DEBTINC': 'Debt-to-income ratio'}
        names = [feat_dict[i] for i in names]
        colors = [pos_color if x > 0 else neg_color for x in vals]
        pos = np.arange(len(vals)) + .5

        ax.barh(pos, vals, align = 'center', color = colors)
        ax.set_yticks(pos)
        ax.set_yticklabels(names)

        

        return fig
    
    plt.rcParams['font.family'] = 'Segoe UI'
    
    exp = lime_explainer.explain_instance(
                                        data_row = input,
                                        predict_fn = model.predict_proba
                                        )

    fig = custom_lime_plot(exp)
    fig.set_size_inches(11, 9)

    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel('Feature', fontsize = 21, color = '#6c6c6c')
    ax.set_xlabel('Contribution', fontsize = 21, color = '#6c6c6c')
    ax.tick_params(axis='x', labelsize = 20)
    ax.tick_params(axis='y', labelsize = 20)

    plt.title('')
    for xticklabel, yticklabel in zip(ax.get_xticklabels(),ax.get_yticklabels()):
            xticklabel.set_color('#6c6c6c')
            yticklabel.set_color('#6c6c6c')

    fig.savefig(os.path.join(current_dir, 'static', 'lime_explanation.png'),
                format = 'png', bbox_inches = 'tight', dpi = 300, transparent = True)
    plt.close(fig)

@app.route('/')
def home():  
    features = inputs['final_features']
    categorical_features = inputs['categorical_features']
    return render_template('index.html', variables = features,
                           categorical_features = categorical_features)


@app.route('/predict', methods = ['POST'])
def predict():
    woe_bins, woe_binning, model, threshold, final_features, categorical_features, input_df, *_ = (input for _, input in inputs.items())

    for feature in input_df.columns:
        if feature in final_features:
            if feature in categorical_features:
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

    lime_plot(input_df_woe_FINAL.iloc[0], model, lime_explainer)

    return render_template('results.html',
                           prediction = round(pred_score[0] * 100, 2),
                           predicted_class = result[0])


if __name__ == '__main__':
    app.run(debug = True)