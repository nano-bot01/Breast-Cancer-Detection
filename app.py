# """
# Created on Sun Mar 26 22:15:42 2023

# @author: ankit
# """

# import numpy as np
# import pandas as pd
# import pickle
# from flask import Flask, request, render_template

# app = Flask(__name__)

import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

model = pickle.load(open("breast_cancer_detector.pickle", 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

# @app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['GET', 'POST']) 
def predict():
    if request.method == 'POST':
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
                         'worst concave points', 'worst symmetry', 'worst fractal dimension']

        df = pd.DataFrame(feature_values, columns=feature_names)

        output = model.predict(df)

        if output == 1:
            result = "---Patient has Breast Cancer---"
        else:
            result = "---Patient is Healthy---"

        return render_template('index.html', prediction_text='The Result is : {} '.format(result))

if __name__ == "__main__":
    app.run()

