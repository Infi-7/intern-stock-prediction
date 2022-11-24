import pandas as pd
import os
import pickle
from flask import *

from io import *
from flask import Flask
import matplotlib.pyplot as plt
import urllib
import base64

#print(os.getcwd())

os.chdir('C:\\Users\\infip\\OneDrive\\Documents\\PY Projects\\not ready\\stock prediction\\')


with open('model_stock_predict','rb') as f:
    stocks_pred = pickle.load(f)

df =pd.read_csv("prices.csv", header=0)
comp_info = pd.read_csv("securities.csv")

rows= []
data_names = []
list1 = []
headings = ("Date","Symbol","Open","Close","Low","High","Volume","Close Predictions")

app = Flask(__name__)


@app.route('/',methods=['GET','POST'])
def index():
    con_names = comp_info["Security"].unique()
    for a_l in con_names:
        data_names.append(a_l)

    return render_template('index.html',data_names=data_names)



@app.route('/predict',methods=['GET','POST'])
def predict():
    select = request.form.get('selected_names')
    df2 = comp_info.iloc[:, [0, 1]]
    area_dict = dict(zip(df2.Security, df2.Ticker_symbol))
    ticker_value = area_dict[select]
    results= df[df.symbol.isin ([ticker_value])]
    # Update the data frame starting with 2nd records , since first prediction is for 2nd record
    results = results[2:]
    # Reset the index 0, 1,2 etc
    results = results.reset_index(drop=True)
    # Convert Predicted Value to Data Frame
    df_stocks_pred = pd.DataFrame(stocks_pred, columns=['Close_Prediction'])
    # Concat Original and prediction data
    results = pd.concat([results, df_stocks_pred], axis=1)
    final_results = results.head(15)

    rows = final_results.values.tolist()


    img = BytesIO()
    plt.rcParams["figure.figsize"] = (9, 7)
    plt.plot(results['close'], 'b',label='Actual Closing Price')
    plt.plot(results['Close_Prediction'], 'r',label='Predicted Closing Price')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Stock Prices')
    plt.title('Check the accuracy of the model with time')
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_data = urllib.parse.quote(base64.b64encode(img.getvalue()).decode('utf-8'))

    return render_template('predict.html', headings=headings,rows=rows, plot_url=plot_data)

if __name__ == '__main__':
    app.run(port = 5000, debug = True)