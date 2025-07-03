
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load('model/model.pkl')
preprocessor = joblib.load('model/preprocessor.pkl')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            data = {
                'carat': float(request.form['carat']),
                'depth': float(request.form['depth']),
                'table': float(request.form['table']),
                'x': float(request.form['x']),
                'y': float(request.form['y']),
                'z': float(request.form['z']),
                'cut': request.form['cut'],
                'color': request.form['color'],
                'clarity': request.form['clarity']
            }
            df = pd.DataFrame([data])
            transformed = preprocessor.transform(df)
            prediction = model.predict(transformed)[0]
            return render_template('index.html', results=round(prediction, 2))
        except Exception as e:
            return render_template('index.html', results=str(e))
    return render_template('index.html', results=None)

if __name__ == '__main__':
    app.run(debug=True)
