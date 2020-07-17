from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

cols_when_model_builds = model.get_booster().feature_names

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    for rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = pd.DataFrame(float_features).transpose()
    final_features.columns = [cols_when_model_builds]
    #final_features.columns = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'key',
     #                     'liveness', 'loudness', 'mode', 'speechiness', 'tempo', 'valence']
    prediction = model.predict(final_features)



    return render_template('index.html', prediction_text=prediction)

#@app.route('/predict_api', methods=['POST'])
#def predict_api():
 #   '''
  #  for direct API calls
   # '''
    #data = request.get_json(force=True)
    #prediction = model.predict([np.array(list(data.values()))])

    #output = prediction[0]
    #return jsonify(output)


app.run('127.0.0.1', debug=True)