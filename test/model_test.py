#predict new data
import numpy as np
import pandas

test_data = np.array([[38,"Private",215646,"HS-grad",9,"Divorced","Handlers-cleaners","Not-in-family","White","Male",0,0,40,"United-States"]])
test_df = pandas.DataFrame(test_data, columns = ["age","workclass","fnlwgt","education","educational-num","marital-status","occupation","relationship","race","gender","capital-gain","capital-loss","hours-per-week","native-country"])

#Load trained model
import joblib
model_folder = '../model/'
filePath = model_folder + 'best_model.pkl'

#open file
file = open(filePath, "rb")
#load the trained model
trained_model = joblib.load(file)

#Predict with trained model
prediction = trained_model.predict(test_df)
print(prediction)