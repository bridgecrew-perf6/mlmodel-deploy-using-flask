#import Flask
from flask import Flask, render_template, request
import numpy as np
import pandas
import joblib

#load  the model from the specified location.
model_folder = 'models/'
filePath = model_folder + 'best_model.pkl'
file = open(filePath, "rb")
loaded_model = joblib.load(file)


#create an instance of Flask
app = Flask(__name__)


#By default this home.html will be loaded from the templates folder when we hit the http://127.0.0.1:5000
@app.route('/')
def home():
    return render_template('home.html')

# prediction function that makes use of the trained ML model to make predictions on the new data points provided.
def ValuePredictor(to_predict_list):
    """
    take the new input parameters as list and return prediction made from the loaded model.
    :param to_predict_list: list of new input parameters from the end user.
    :return: salary prediction made by the model.
    """
    to_predict = np.array(to_predict_list)
    test_df = pandas.DataFrame(to_predict, columns = ["age","workclass","education","educational-num","marital-status","occupation","relationship","race","gender","capital-gain","capital-loss","hours-per-week","native-country"])
    result = loaded_model.predict(test_df)
    return result[0]

# This route will return the result of the prediction made.
@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())

        #convert the values to int
        #in production app you will have checks to make sure inputs are of correct type
        to_predict_list[0]=int(to_predict_list[0])
        to_predict_list[3]=int(to_predict_list[3])
        to_predict_list[9]=int(to_predict_list[9])
        to_predict_list[10]=int(to_predict_list[10])
        to_predict_list[11]=int(to_predict_list[11])

        to_predict_list = [to_predict_list]

        result = ValuePredictor(to_predict_list)       
        if result== "<=50K":
            prediction ='Income less than 50K'
        else:
            prediction ='Income more that 50K'           
        return render_template("result.html", prediction = prediction)


if __name__ == '__main__':
    app.run(port=5000, debug=True)