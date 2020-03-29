import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('user-symptoms.html')



def ValuePredictor(to_predict_list): 
    to_predict = np.array(to_predict_list).reshape(1,7) 
    loaded_model = pickle.load(open("model.pkl", "rb")) 
    result = loaded_model.predict_proba(to_predict) 
    return result[0] 
  
@app.route('/result', methods = ['POST']) 
def result(): 
    if request.method == 'POST': 
        to_predict_list = request.form.to_dict() 
        to_predict_list = list(to_predict_list.values()) 
        to_predict_list = list(map(int, to_predict_list)) 
        print(to_predict_list)
        result = ValuePredictor(to_predict_list)         
        if(result[0] >= result[1]): 
            prediction =result[0]
            
        else: 
            prediction =result[1]
        return render_template("result.html",prediction = 'Chances of corona positive is  {} '.format(prediction)) 

if __name__ == '__main__':
    app.run(debug=True)
