import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from flask import *
from flask_cors import CORS, cross_origin
import os
from prophet import Prophet
from waitress import serve
 
UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
 
# Define allowed files
ALLOWED_EXTENSIONS = {'csv'}
 
app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
app.secret_key = 'This is your secret key to utilize session in Flask'
 
 
@app.route('/', methods=['POST'])
def uploadFile():
    if request.method == 'POST':
      # upload file flask
        f = request.files.get('file')

        filename = f.filename
        file_extension = filename.split('.')[1]

        if(file_extension != "csv" and file_extension != "xlsx"):
            return {
                "status": "failed",
                "msg":"Invalid file type. Please select again !"
            }
        f.save(os.path.join(app.config['UPLOAD_FOLDER'],filename))
 
        session['uploaded_data_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        df = pd.read_csv(session['uploaded_data_file_path'])
        result = {
            "columns": list(df.columns)
        }
        
        return result
@app.route('/predict', methods=['POST'])
def getPrediction():
    data = request.get_json()

    if data:
        filename = data["filename"]

        file_extension = filename.split('.')[1]

        num_future_ds = data["furtureDs"]
        model = data["model"]
        ds_col = data["dsCol"]
        y_col = data["yCol"]
        date_format = data["dateFormat"]

        if (file_extension == "csv"):
            df = pd.read_csv(f"./staticFiles/uploads/{filename}")
        else:
            df = pd.read_excel(f"./staticFiles/uploads/{filename}")

        df = df[[ds_col, y_col]].copy()
        df = df.iloc[-365:,:].copy()


        df.columns = ['ds', 'y']
        try:
            df['ds'] = pd.to_datetime(df['ds'], format=date_format)
        except:
            return {
                "status": "failed",
                "msg":"Format of \"Date\" column is invalid. Please select again !"
            }
        if is_numeric_dtype(df['y']) == False:
            return {
                "status": "failed",
                "msg":"Values of \"y\" column is invalid. Please select again !"
            }
        model = Prophet()
        model.fit(df)
        x_valid = model.make_future_dataframe(periods=num_future_ds, freq="D", include_history=False)
        y_pred = model.predict(x_valid)


        result_ds = list(df["ds"])
        result_y = list(df["y"])
        result_predicted_ds = list(x_valid["ds"])
        result_predicted_y = list(y_pred["yhat"])
        result_ds.append(result_predicted_ds[0])
        result_y.append(result_predicted_y[0])


        result = {
            "status": "success",
            "msg":"Model has been built successfully.",
            "data": {
                "ds": result_ds,
                "y": result_y,
                "predicted_ds": result_predicted_ds,
                "predicted_y": result_predicted_y,
            }
        }
        return result

    return "Filename is empty"
 
 

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port='7777')
    serve(app, host='0.0.0.0', port='7777', threads = 1)