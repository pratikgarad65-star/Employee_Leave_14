import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "naive_bayes.pkl"
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Categorical mappings (Standard Employee Dataset Label Encodings)
EDUCATION_MAP = {"Bachelors": 0, "Masters": 1, "PHD": 2}
CITY_MAP = {"Bangalore": 0, "Pune": 1, "New Delhi": 2}
GENDER_MAP = {"Female": 0, "Male": 1}
BENCHED_MAP = {"No": 0, "Yes": 1}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Employee Leave Prediction | ML Dashboard</title>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #311042 100%);
      --card-bg: rgba(255, 255, 255, 0.05);
      --card-border: rgba(255, 255, 255, 0.12);
      --accent: #818cf8;
      --accent-hover: #6366f1;
      --text-main: #f8fafc;
      --text-dim: #94a3b8;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Plus Jakarta Sans', sans-serif; }
    
    body {
      background: var(--bg-gradient);
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--text-main);
      padding: 2rem 1rem;
    }

    .container {
      width: 100%;
      max-width: 820px;
      background: var(--card-bg);
      backdrop-filter: blur(18px);
      -webkit-backdrop-filter: blur(18px);
      border: 1px solid var(--card-border);
      border-radius: 24px;
      padding: 2.5rem;
      box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    }

    .header {
      text-align: center;
      margin-bottom: 2rem;
    }

    .header h1 {
      font-size: 2rem;
      font-weight: 700;
      letter-spacing: -0.02em;
      background: linear-gradient(to right, #e0e7ff, #c084fc);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 0.5rem;
    }

    .header p {
      color: var(--text-dim);
      font-size: 0.95rem;
    }

    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 1.25rem;
    }

    .form-group {
      display: flex;
      flex-direction: column;
      gap: 0.45rem;
    }

    label {
      font-size: 0.825rem;
      font-weight: 600;
      color: var(--text-dim);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }

    input, select {
      background: rgba(15, 23, 42, 0.6);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 0.75rem 1rem;
      color: #fff;
      font-size: 0.95rem;
      transition: all 0.2s ease;
      outline: none;
    }

    input:focus, select:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(129, 140, 248, 0.25);
    }

    select option {
      background: #1e1b4b;
      color: #fff;
    }

    .btn-submit {
      grid-column: 1 / -1;
      margin-top: 1rem;
      background: linear-gradient(135deg, #6366f1, #a855f7);
      color: #fff;
      border: none;
      border-radius: 12px;
      padding: 1rem;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      box-shadow: 0 10px 20px -5px rgba(99, 102, 241, 0.5);
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 0.5rem;
    }

    .btn-submit:hover {
      transform: translateY(-2px);
      box-shadow: 0 15px 25px -5px rgba(99, 102, 241, 0.65);
    }

    .result-box {
      margin-top: 2rem;
      padding: 1.25rem;
      border-radius: 16px;
      display: none;
      text-align: center;
      animation: fadeIn 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .result-box.leave {
      background: rgba(239, 68, 68, 0.15);
      border: 1px solid rgba(239, 68, 68, 0.4);
      color: #fca5a5;
    }

    .result-box.stay {
      background: rgba(34, 197, 94, 0.15);
      border: 1px solid rgba(34, 197, 94, 0.4);
      color: #86efac;
    }

    .result-title {
      font-size: 1.15rem;
      font-weight: 700;
    }

    .result-subtitle {
      font-size: 0.85rem;
      opacity: 0.8;
      margin-top: 0.25rem;
    }
  </style>
</head>
<body>

  <div class="container">
    <div class="header">
      <h1>Employee Turnover Predictor</h1>
      <p>Gaussian Naive Bayes Model Inference Engine</p>
    </div>

    <form id="predictionForm" class="grid">
      <div class="form-group">
        <label>Education</label>
        <select name="Education" required>
          <option value="Bachelors">Bachelors</option>
          <option value="Masters">Masters</option>
          <option value="PHD">PHD</option>
        </select>
      </div>

      <div class="form-group">
        <label>Joining Year</label>
        <input type="number" name="JoiningYear" value="2017" min="2000" max="2030" required />
      </div>

      <div class="form-group">
        <label>City</label>
        <select name="City" required>
          <option value="Bangalore">Bangalore</option>
          <option value="Pune">Pune</option>
          <option value="New Delhi">New Delhi</option>
        </select>
      </div>

      <div class="form-group">
        <label>Payment Tier</label>
        <select name="PaymentTier" required>
          <option value="1">Tier 1 (Highest)</option>
          <option value="2">Tier 2</option>
          <option value="3" selected>Tier 3</option>
        </select>
      </div>

      <div class="form-group">
        <label>Age</label>
        <input type="number" name="Age" value="28" min="18" max="70" required />
      </div>

      <div class="form-group">
        <label>Gender</label>
        <select name="Gender" required>
          <option value="Male">Male</option>
          <option value="Female">Female</option>
        </select>
      </div>

      <div class="form-group">
        <label>Ever Benched</label>
        <select name="EverBenched" required>
          <option value="No">No</option>
          <option value="Yes">Yes</option>
        </select>
      </div>

      <div class="form-group">
        <label>Experience in Domain (Years)</label>
        <input type="number" name="ExperienceInCurrentDomain" value="3" min="0" max="40" required />
      </div>

      <button type="submit" class="btn-submit" id="submitBtn">Run Prediction</button>
    </form>

    <div id="resultBox" class="result-box">
      <div class="result-title" id="resultTitle"></div>
      <div class="result-subtitle" id="resultSubtitle"></div>
    </div>
  </div>

  <script>
    const form = document.getElementById('predictionForm');
    const resultBox = document.getElementById('resultBox');
    const resultTitle = document.getElementById('resultTitle');
    const resultSubtitle = document.getElementById('resultSubtitle');
    const submitBtn = document.getElementById('submitBtn');

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      submitBtn.innerText = 'Calculating...';
      submitBtn.disabled = true;

      const formData = new FormData(form);
      const data = Object.fromEntries(formData.entries());

      try {
        const response = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(data)
        });

        const res = await response.json();
        resultBox.style.display = 'block';

        if (res.error) {
          resultBox.className = 'result-box leave';
          resultTitle.innerText = 'Error';
          resultSubtitle.innerText = res.error;
        } else {
          if (res.prediction === 1) {
            resultBox.className = 'result-box leave';
            resultTitle.innerText = 'High Risk of Departure';
            resultSubtitle.innerText = `Probability to Leave: ${(res.probability[1] * 100).toFixed(1)}%`;
          } else {
            resultBox.className = 'result-box stay';
            resultTitle.innerText = 'Likely to Stay';
            resultSubtitle.innerText = `Probability to Stay: ${(res.probability[0] * 100).toFixed(1)}%`;
          }
        }
      } catch (err) {
        resultBox.style.display = 'block';
        resultBox.className = 'result-box leave';
        resultTitle.innerText = 'Prediction Failed';
        resultSubtitle.innerText = 'Check server logs or connection.';
      } finally {
        submitBtn.innerText = 'Run Prediction';
        submitBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json()

        # Feature mapping and validation
        education = EDUCATION_MAP.get(payload.get("Education"), 0)
        joining_year = int(payload.get("JoiningYear", 2017))
        city = CITY_MAP.get(payload.get("City"), 0)
        payment_tier = int(payload.get("PaymentTier", 3))
        age = int(payload.get("Age", 28))
        gender = GENDER_MAP.get(payload.get("Gender"), 1)
        ever_benched = BENCHED_MAP.get(payload.get("EverBenched"), 0)
        experience = int(payload.get("ExperienceInCurrentDomain", 1))

        # Array ordered strictly matching model.feature_names_in_
        features = np.array([[
            education,
            joining_year,
            city,
            payment_tier,
            age,
            gender,
            ever_benched,
            experience
        ]])

        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0].tolist()

        return jsonify({
            "status": "success",
            "prediction": int(prediction),
            "probability": probabilities
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
