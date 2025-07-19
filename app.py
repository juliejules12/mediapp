from flask import Flask, render_template, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from datetime import datetime

app = Flask(__name__)

def generate_health_data():
    timestamps = pd.date_range(end=datetime.now(), periods=100, freq='T')
    heart_rate = 70 + 10 * np.sin(np.linspace(0, 3 * np.pi, 100)) + np.random.randint(-5, 5, 100)
    blood_oxygen = 95 + np.random.normal(0, 1, 100)
    activity_level = np.random.choice(['low', 'moderate', 'high'], 100)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'heart_rate': heart_rate,
        'blood_oxygen': blood_oxygen,
        'activity_level': activity_level
    })

    return df

def detect_anomalies(df):
    model = IsolationForest(contamination=0.1, random_state=42)
    df['anomaly'] = model.fit_predict(df[['heart_rate', 'blood_oxygen']])
    df['status'] = df['anomaly'].apply(lambda x: 'Anomaly' if x == -1 else 'Normal')
    return df

@app.route('/')
def home():
    df = generate_health_data()
    df = detect_anomalies(df)
    latest = df.iloc[-1]
    return render_template('index.html', data=latest.to_dict())

@app.route('/data')
def data():
    df = generate_health_data()
    df = detect_anomalies(df)
    return jsonify(df.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
