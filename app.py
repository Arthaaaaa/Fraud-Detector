from flask import Flask, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model + scaler
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

HTML = """
<!doctype html>
<html lang="id">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>🔍 Fraud Detector - ANN Model</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    * {{
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }}
    
    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      min-height: 100vh;
      padding: 20px;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
    }}
    
    .container {{
      max-width: 600px;
      width: 100%;
      animation: fadeInUp 0.6s ease-out;
    }}
    
    @keyframes fadeInUp {{
      from {{
        opacity: 0;
        transform: translateY(30px);
      }}
      to {{
        opacity: 1;
        transform: translateY(0);
      }}
    }}
    
    .header {{
      text-align: center;
      color: white;
      margin-bottom: 30px;
    }}
    
    .header h1 {{
      font-size: 2.5rem;
      font-weight: 700;
      margin-bottom: 10px;
      text-shadow: 0 2px 10px rgba(0,0,0,0.2);
      letter-spacing: -0.5px;
    }}
    
    .header .subtitle {{
      font-size: 1rem;
      font-weight: 300;
      opacity: 0.95;
    }}
    
    .card {{
      background: white;
      border-radius: 20px;
      padding: 35px;
      box-shadow: 0 20px 60px rgba(0,0,0,0.3);
      backdrop-filter: blur(10px);
    }}
    
    .info-box {{
      background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
      border-left: 4px solid #667eea;
      padding: 15px;
      border-radius: 10px;
      margin-bottom: 25px;
      font-size: 0.9rem;
      color: #2d3748;
      line-height: 1.6;
    }}
    
    .info-box b {{
      color: #667eea;
      font-weight: 600;
    }}
    
    .form-group {{
      margin-bottom: 25px;
    }}
    
    .form-group label {{
      display: block;
      font-weight: 600;
      color: #2d3748;
      margin-bottom: 8px;
      font-size: 0.95rem;
      letter-spacing: 0.3px;
    }}
    
    .form-group input {{
      width: 100%;
      padding: 14px 16px;
      border: 2px solid #e2e8f0;
      border-radius: 12px;
      font-size: 1rem;
      font-family: 'Inter', sans-serif;
      transition: all 0.3s ease;
      background: #f8f9fa;
    }}
    
    .form-group input:focus {{
      outline: none;
      border-color: #667eea;
      background: white;
      box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
      transform: translateY(-2px);
    }}
    
    .form-group input:hover {{
      border-color: #cbd5e0;
    }}
    
    .btn-submit {{
      width: 100%;
      padding: 16px;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      border: none;
      border-radius: 12px;
      font-size: 1.1rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.3s ease;
      box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
      font-family: 'Inter', sans-serif;
      letter-spacing: 0.5px;
      margin-top: 10px;
    }}
    
    .btn-submit:hover {{
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    }}
    
    .btn-submit:active {{
      transform: translateY(0);
    }}
    
    .result-container {{
      margin-top: 25px;
      animation: slideIn 0.4s ease-out;
    }}
    
    @keyframes slideIn {{
      from {{
        opacity: 0;
        transform: translateX(-20px);
      }}
      to {{
        opacity: 1;
        transform: translateX(0);
      }}
    }}
    
    .result-box {{
      padding: 20px;
      border-radius: 12px;
      font-size: 1rem;
      line-height: 1.8;
      box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }}
    
    .result-box.ok {{
      background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
      border-left: 4px solid #28a745;
      color: #155724;
    }}
    
    .result-box.bad {{
      background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
      border-left: 4px solid #dc3545;
      color: #721c24;
    }}
    
    .result-box b {{
      font-weight: 600;
      font-size: 1.1rem;
    }}
    
    .result-box .probability {{
      margin-top: 12px;
      padding-top: 12px;
      border-top: 1px solid rgba(0,0,0,0.1);
      font-size: 0.95rem;
    }}
    
    .icon {{
      font-size: 1.3rem;
      margin-right: 8px;
      vertical-align: middle;
    }}
    
    @media (max-width: 640px) {{
      .header h1 {{
        font-size: 2rem;
      }}
      
      .card {{
        padding: 25px 20px;
      }}
      
      body {{
        padding: 15px;
      }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>🔍 Fraud Detector</h1>
      <p class="subtitle">Artificial Neural Network Model</p>
    </div>
    
    <div class="card">
      <div class="info-box">
        <strong>📊 Model Info:</strong> Input 3 fitur (<b>Time</b>, <b>Amount</b>, <b>V14</b>). 
        Architecture: 1 hidden layer dengan 5 neuron (ReLU activation) + output layer (Sigmoid).
      </div>
      
      <form method="POST" action="/predict">
        <div class="form-group">
          <label>⏱️ Time</label>
          <input name="time" type="number" step="any" required placeholder="Masukkan nilai Time (contoh: 10000)">
        </div>

        <div class="form-group">
          <label>💰 Amount</label>
          <input name="amount" type="number" step="any" required placeholder="Masukkan nilai Amount (contoh: 149.62)">
        </div>

        <div class="form-group">
          <label>📈 V14</label>
          <input name="v14" type="number" step="any" required placeholder="Masukkan nilai V14 (contoh: -2.3)">
        </div>

        <button type="submit" class="btn-submit">🚀 Lakukan Prediksi</button>
      </form>

      <div class="result-container">
        {result_block}
      </div>
    </div>
  </div>
</body>
</html>
"""

@app.get("/")
def home():
    return HTML.format(result_block="")

@app.post("/predict")
def predict():
    try:
        time = float(request.form["time"])
        amount = float(request.form["amount"])
        v14 = float(request.form["v14"])

        X = np.array([[time, amount, v14]], dtype=float)
        X_scaled = scaler.transform(X)

        pred = model.predict(X_scaled)[0]  # 0 atau 1
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_scaled)[0][1]  # peluang class=1

        if pred == 1:
            msg = "<div class='result-box bad'><span class='icon'>⚠️</span><b>Hasil Prediksi:</b> FRAUD DETECTED (1)<br><small style='opacity:0.9;'>Transaksi ini terdeteksi sebagai penipuan</small>"
        else:
            msg = "<div class='result-box ok'><span class='icon'>✅</span><b>Hasil Prediksi:</b> NORMAL (0)<br><small style='opacity:0.9;'>Transaksi ini terlihat normal</small>"

        if proba is not None:
            proba_pct = proba * 100
            msg += f"<div class='probability'><b>📊 Probabilitas Fraud:</b> {proba_pct:.2f}% ({proba:.4f})</div>"

        msg += "</div>"

        return HTML.format(result_block=msg)
    except Exception as e:
        return HTML.format(result_block=f"<div class='result-box bad'><span class='icon'>❌</span><b>Error:</b> {str(e)}</div>")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
