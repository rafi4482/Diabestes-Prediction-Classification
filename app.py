from flask import Flask, render_template, request

# Load the trained model when the Flask app starts
model = joblib.load('model.pkl')

app = Flask(__name__)

# Load your trained model
model = RandomForestClassifier(n_estimators=310, min_samples_split=12, min_samples_leaf=5, max_depth=None)
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get user input
        gender = request.form['gender']
        smoking_history = request.form['smoking_history']
        hba1c_level = float(request.form['hba1c_level'])

        # Create a DataFrame with user input
        user_data = pd.DataFrame({
            'gender': [gender],
            'smoking_history': [smoking_history],
            'HbA1c_level': [hba1c_level]
        })

        # Preprocess the user data (if necessary)

        # Make a prediction using the loaded model
        prediction = model.predict(user_data)

        # Display the prediction result
        return render_template('result.html', prediction=prediction[0])