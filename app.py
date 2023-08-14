import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
from flask import *
from flask_cors import CORS, cross_origin
import os
from prophet import Prophet
import logging
from waitress import serve
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(message)s')

UPLOAD_FOLDER = os.path.join('staticFiles', 'uploads')
 
# Define allowed files
ALLOWED_EXTENSIONS = {'csv'}
 
app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
 
app.secret_key = 'This is your secret key to utilize session in Flask'
 

@app.route('/', methods=['POST'])
def wakeup_call():
    if request.method == 'POST':
      
        result = {
            "status": "success"
        }
        
        return result
@app.route('/upload', methods=['POST'])
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

        if (file_extension == "csv"):
            df = pd.read_csv(session['uploaded_data_file_path'])
        elif (file_extension == "xlsx"):
            df = pd.read_excel(session['uploaded_data_file_path'])
        else:
            return {
                "status": "failed",
                "msg":"Can not open file. Please select again !"
            }
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
        elif (file_extension == "xlsx"):
            df = pd.read_excel(f"./staticFiles/uploads/{filename}")
        else:
            return {
                "status": "failed",
                "msg":"Can not open file. Please select again !"
            }

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
        

        if(model == "prophet"):
            model = Prophet()
            model.fit(df)
            x_valid = model.make_future_dataframe(periods=num_future_ds, freq="D", include_history=False)
            y_pred = model.predict(x_valid)
            result_predicted_ds = list(x_valid["ds"])
            result_predicted_y = list(y_pred["yhat"])
        elif(model == "lstm"):
            x_valid = pd.DataFrame({"ds":pd.date_range(start=df["ds"].iloc[-1], periods=num_future_ds)})
            y = np.array(df["y"].iloc[-365:]).reshape(-1, 1)
            minmax_scaler = MinMaxScaler()
            scaled_y = minmax_scaler.fit_transform(y)
            n_input = 12
            n_features = 1

            generator = TimeseriesGenerator(scaled_y, scaled_y, length=n_input, batch_size=1)
            model = Sequential()
            model.add(LSTM(units = 128, input_shape=(n_input, n_features), return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(units = 128, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(units = 128, return_sequences=True))
            model.add(Dropout(0.2))
            model.add( LSTM(units=128,  return_sequences=False))
            model.add(Dropout(0.2))
            model.add(Dense(units=1, activation = "swish"))
            model.compile(optimizer='adam', loss='mse')
            model.fit(generator, epochs = 50, batch_size = 32)


            test_predictions = []

            first_eval_batch = scaled_y[-n_input:]
            current_batch = first_eval_batch.reshape((1, n_input, n_features))

            for i in range(num_future_ds):

                current_pred = model.predict(current_batch)[0]
                test_predictions.append(current_pred)
                current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1)
            test_predictions = test_predictions[1:]
            y_pred = minmax_scaler.inverse_transform(test_predictions)
            result_predicted_ds = list(x_valid["ds"])
            result_predicted_y = list(np.squeeze(y_pred, axis=(1, )))

        # elif(model == "lstm_gru"):

        else:
            return {"status": "failed","msg":"Model name is wrong"}


        result_ds = list(df["ds"])
        result_y = list(df["y"])
        
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