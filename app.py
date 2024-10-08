from flask import Flask, render_template, request, url_for, redirect
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.mlproject.exception import CustomException
from src.mlproject.pipelines.prediction_pipeline import PredictPipeline, CustomData

application = Flask(__name__)

app = application

# route for homepage
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get("writing_score"))
        )

        predict_df = data.get_data_as_data_frame()
        print(predict_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(predict_df)
        
        # Redirect to results page
        return redirect(url_for('result', results=results[0]))

@app.route('/result', methods=['GET'])
def result():
    results = request.args.get('results')
    return render_template('result.html', results=results)

if __name__ == "__main__":
    app.run(debug=True)
