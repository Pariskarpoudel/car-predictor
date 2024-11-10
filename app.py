from flask import Flask , render_template,request
import pandas as pd 
import json
import pickle
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer 
from flask_cors import CORS,cross_origin

app = Flask(__name__)
cors=CORS(app)

car = pd.read_csv('Cleaned_car.csv')

with open('model.pkl', 'rb') as file:
        pipe = pickle.load(file)



@app.route('/')
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(),reverse=True)
    fuel_type = car['fuel_type'].unique()
    car_models_json = json.dumps(car_models)
    return render_template('index.html', companies=companies, car_models=car_models_json, years=year, fuel_types=fuel_type)


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        # Get form data
        company = request.form.get('company')
        car_model = request.form.get('car_models')
        year = int(request.form.get('year'))
        fuel_type = request.form.get('fuel_type')
        driven = int(request.form.get('kilo_driven'))

        # Create input data as a list (similar to how you might have trained it)
        input_list = [[car_model, company, year, driven, fuel_type]]

        input_data = pd.DataFrame(input_list, 
                                columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])
        


        # Debug prints
        print("Input DataFrame:")
        print(input_data.dtypes)
        print(input_data)
        # Load pipeline again to ensure it's fresh
        with open('model.pkl', 'rb') as f:
            pipe = pickle.load(f)

        print("\nPipeline steps:", pipe.named_steps.keys())
        # Make prediction
        pred = pipe.predict(input_data)
        
        return str(np.round(pred[0], 2))
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return str(e), 400

# warning vandai dekhairathyo u are using 1.5.2 instead of 1.2.2 , but i ignored it that was problem , now i installed 1.2.2 and corresponding compatiable versinos of numpy and scipy 

if __name__ == '__main__':
    app.run(debug=True)