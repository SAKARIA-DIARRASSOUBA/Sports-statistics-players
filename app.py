from flask import Flask, render_template,request
import pandas as pd
import numpy as np
import joblib 

with open('Classifier','rb') as file:

    model=joblib.load(file)

with open("MinMaxScalers","rb") as f:
    normalizer=joblib.load(f)


app = Flask(__name__)

@app.route('/',methods=['GET'])
def home():
	return render_template('main.html')

decision=[ "Bad to reversing","Good to reversing"]

@app.route('/', methods=['POST','GET'])
def main(): 
    if request.method == 'GET':
        return (flask.render_template('main.html'))

    if request.method == 'POST':
        
        data=[float(x) for x in request.form.values()]
        array_features = np.array([data])
        array_features.reshape(-1,1)
        normalize_features=normalizer.transform(array_features)
        prediction = model.predict(normalize_features)
        output = round(prediction[0])
        
       
    return render_template('main.html', 
         original_input={'GP':data[0],'PTS':data[1],'FGPer':data[2],'3P_Made':data[3], 
         '3PPre':data[4],'FTM':data[5],'FTPer':data[6],'REB':data[7],
         'AST':data[8],'STL':data[9],'BLK':data[10],'TOV':data[11]} ,
        result=decision[output] )


                              


if __name__ == '__main__':
    app.run(debug=True)

