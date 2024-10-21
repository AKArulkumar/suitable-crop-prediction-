from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained decision tree model
model = pickle.load(open('model/trained_crop_model.pkl', 'rb'))

# Load the dataset for reference
data = pd.read_csv('dataset/tamilnadu_crop_data_with_suitability.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    district = request.form['district']
    season = request.form['season']
    crop = request.form['crop']
    soil = request.form['soil']

    # Filter dataset based on district and crop
    suitable_data = data[(data['District'] == district) & (data['Crop'] == crop)]

    if suitable_data.empty:
        suitability_message = "The crop is not suitable for the selected district."
        return render_template('result.html', crop=crop, soil=soil, temperature="N/A", rainfall="N/A", suitability_message=suitability_message)

    # Check if the soil type matches
    suitable_soil = suitable_data[suitable_data['Soil Type'] == soil]

    if suitable_soil.empty:
        suitability_message = "This crop is not suitable for the selected soil type."
    else:
        # Fetch details
        temperature = suitable_soil.iloc[0]['Average Temperature (°C)']
        rainfall = suitable_soil.iloc[0]['Average Rainfall (mm)']
        if season != suitable_soil.iloc[0]['Ideal Season']:
            suitability_message = "This season is not ideal for this crop."
        else:
            suitability_message = "This crop is suitable for the selected soil and season."

        return render_template('result.html', crop=crop, soil=soil, temperature=temperature, rainfall=rainfall, suitability_message=suitability_message)
