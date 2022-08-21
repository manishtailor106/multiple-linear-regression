


import numpy as np
from flask import Flask, request, jsonify, render_template

import pickle


app = Flask(__name__)
model = pickle.load(open('house-prices2.pkl','rb')) 


@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    
    '''
    For rendering results on HTML GUI
    '''
    #result=np.array(['SqFt','bedrooms','Offers','bricks','Neighborhood','Bathrooms'])
    #result.reshape(1,-1)
    #print(result)
    r1 = float(request.args.get('SqFt'))
    r2 = float(request.args.get('bedrooms'))
    r3 = float(request.args.get('Offers'))
    r4 = float(request.args.get('bricks'))
    r5 = float(request.args.get('Neighborhood'))
    r6 = float(request.args.get('Bathrooms'))
    result=np.array([r1,r2,r3,r4,r5,r6]).reshape(1,-1)
    prediction = model.predict(result)
    
        
    return render_template('index.html', prediction_text='Regression Model  has predicted price for given Square feet is : {}'.format(prediction))



if __name__ == "__main__":
    app.run(debug=True)
