import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
data = pd.read_csv('Clean_Bengaluru_House_Data.csv')
pipe = pickle.load(open("model.pkl",'rb'))

@app.route('/')

def index():

    locations = sorted(data['location'].unique())
    availability = sorted(data['availability'].unique())
    return render_template('index.html', locations=locations, availability=availability)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('size')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')
    availability = request.form.get('availability')
    input = pd.DataFrame([[availability, location, bhk, sqft, bath]]
                         ,columns=['availability', 'location', 'size', 'total_sqft', 'bath'])
    prediction = pipe.predict(input)[0]
    return str(np.round(prediction,2))

if __name__ == "__main__":
    app.run(debug=True)

#apply port number to run on local machine