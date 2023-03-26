"""
Created on Sun Mar 26 22:15:42 2023

@author: ankit
"""

import numpy as np
import pandas as pd
import pickle
import flask as Flask, request, render_template

app = Flask(__name__)
# The __name__ parameter specifies the name of the current Python module, 
# which Flask uses to locate files relative to the application package. 
# In this case, it creates an instance of the Flask class and assigns it to the variable app.

model = pickle.load(open("breast_cancer_detector.pkl", 'rb'))
# loading the model

# rendering the file 
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    feature_values = [np.array(input_features)]
    
    feature_names = ['mean radius', 'mean texture', 'mean perimeter', 'mean area',
       'mean smoothness', 'mean compactness', 'mean concavity',
       'mean concave points', 'mean symmetry', 'mean fractal dimension',
       'radius error', 'texture error', 'perimeter error', 'area error',
       'smoothness error', 'compactness error', 'concavity error',
       'concave points error', 'symmetry error', 'fractal dimension error',
       'worst radius', 'worst texture', 'worst perimeter', 'worst area',
       'worst smoothness', 'worst compactness', 'worst concavity',
       'worst concave points', 'worst symmetry', 'worst fractal dimension',
       'target']
    # features names is given 
    
    df = pd.DataFrame(feature_values, columns = feature_names)
    # data frame is created
    
    output = model.predict(df)
    # predicting output using ml model
    
    if output == 0:
        result = "---Patient has Breast Cancer---"
    else:
        result = "---Patient is Healthy---"
        
    return render_template('index.htmml', prediction_text = 'The Result is : {} '.format(result))
    # return template with result

if __name__ == "__main__":
    app.run()
    





