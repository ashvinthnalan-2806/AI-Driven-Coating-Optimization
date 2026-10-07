# AI-Driven Coating Formulation & Property Optimization (Web Application)

A professional, responsive web application for predicting coating film properties from chemical formulation and baking process parameters using a trained multi-output Machine Learning model.

---

## 1. Project Overview & Workflow

The application enables polymer chemists and coating engineers to evaluate coating performance through a clean, scientific web interface. It bypasses slow, expensive wet-lab trial-and-error by providing instantaneous predictions backed by a trained machine learning pipeline.

```
Browser (User enters formulation & bake parameters)
       ↓
JavaScript Client Validation & Formulation Mass Balance
       ↓
POST /predict (JSON Payload)
       ↓
Flask REST API Backend (app.py)
       ↓
Strict Feature Ordering [Resin, Pigment, Solvent, Additive, Curing Agent, Temp, Time]
       ↓
Trained Preprocessing Pipeline / ColumnTransformer
       ↓
Trained ML Model (Multi-Output Regressor)
       ↓
Predicted Coating Properties [Hardness, Adhesion, Flexibility, Gloss, Drying Time]
       ↓
JSON Response
       ↓
Interactive Dashboard (Result Cards, Chart.js Profile, Summary Table, History, CSV Export)
```

---

## 2. Key Features

- **No Hard-Coded Formulas:** Every single property estimate is calculated directly by the trained Scikit-learn model via `model.predict()`.
- **Pre-Loaded Architecture:** The model and preprocessing pipelines are loaded into memory once during Flask startup.
- **Dynamic Formulation Mass Indicator:** Live calculation of total composition ($\sum \text{wt}\%$) with color-coded alerts and an optional strict 100% constraint toggle.
- **Interactive Metric Dashboard:** Five distinct result cards displaying calculated property magnitudes with their standard units.
- **Visual Property Magnitude Profile:** Interactive Chart.js bar chart displaying the property profile.
- **Side-by-Side Input Summary:** Formatted table showing input parameters next to the output predictions.
- **Local Prediction History:** Tracks recent session predictions in `localStorage` without requiring an external database.
- **CSV Export:** Download the current formulation recipe and AI predicted properties as a CSV file.
- **Zero-Downtime Health Check:** Automatic `GET /health` polling displaying live model readiness status.

---

## 3. Folder Structure

```
AI_Coating_Web/
│
├── app.py                         # Flask REST API backend server
│
├── models/
│   ├── coating_model.joblib       # Primary trained ML model pipeline
│   ├── preprocessing_pipeline.joblib # Feature scaling pipeline (StandardScaler)
│   └── model_metadata.json        # Model architecture & evaluation metrics
│
├── templates/
│   └── index.html                 # Semantic HTML5 dashboard template
│
├── static/
│   ├── style.css                  # Professional scientific CSS stylesheet
│   └── script.js                  # Frontend controller & Chart.js integration
│
├── data/
│   └── coating_data.csv           # Reference training dataset
│
├── generate_demo_model.py         # Standalone demonstration model generator
├── requirements.txt               # Pinned Python package dependencies
└── README.md                      # Complete application documentation
```

---

## 4. Installation & Local Setup

### Step 1: Open PowerShell or Command Prompt
Navigate to the `AI_Coating_Web` directory:

```powershell
cd C:\Users\Administrator\Desktop\project001\AI_Coating_Web
```

### Step 2: Create a Virtual Environment
```powershell
python -m venv venv
```

### Step 3: Activate the Virtual Environment
On Windows:
```powershell
.\venv\Scripts\activate
```
*(On Linux / macOS: `source venv/bin/activate`)*

### Step 4: Install Dependencies
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 5. How to Run the Web Application

### Step 1: Start the Flask Server
```powershell
python app.py
```

You will see terminal output similar to:
```text
2026-10-04 12:40:00 [INFO] Loaded trained ML model successfully from: .../models/coating_model.joblib
2026-10-04 12:40:00 [INFO] Starting Coating Prediction Server on http://localhost:5000
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on all addresses (0.0.0.0)
 * Running on http://127.0.0.1:5000
```

### Step 2: Open in Your Browser
Open your browser and navigate to:
```text
http://localhost:5000
```

---

### Step 3: (Optional) Share via Global Public HTTPS Link
To share the web application with evaluators, professors, or team members outside your local network without port forwarding:

1. Double-click `launch_global_tunnel.bat` (or run `cloudflared tunnel --url http://127.0.0.1:5000` in a terminal).
2. Copy the generated `https://*.trycloudflare.com` URL.
3. Anyone in the world can access the interactive dashboard and live ML predictions through this link while the tunnel is running.

---

## 6. How to Use the Web Application

1. **Verify Backend Status:** Check the top-right badge. It will indicate **`AI Model Ready ✓`** in green.
2. **Configure Formulation Components:**
   - Resin Content (wt%)
   - Pigment Content (wt%)
   - Solvent Content (wt%)
   - Additive Content (wt%)
   - Curing Agent Content (wt%)
   - Check the **Total Composition** bar to ensure your components sum to $\approx 100.0\text{ wt}\%$.
3. **Configure Baking Parameters:**
   - Curing Temperature (°C)
   - Curing Time (minutes)
4. **Click "PREDICT COATING PROPERTIES":**
   - The button shows a loading spinner: *"Running AI prediction..."*.
   - Within milliseconds, the result panel renders:
     - 5 Metric Cards: **Hardness**, **Adhesion**, **Flexibility**, **Gloss**, and **Drying Time**.
     - An interactive Chart.js bar chart.
     - An Input Formulation Summary table.
     - Adds a timestamped record to the **Prediction History** table.
5. **Download CSV:** Click **"DOWNLOAD RESULTS (CSV)"** to save your recipe and predictions locally.
6. **Reset:** Click **"RESET"** to restore default inputs and clear current result cards.

---

## 7. REST API Documentation

### Endpoint: `POST /predict`
Generates property predictions for a given coating recipe.

#### Request Headers:
`Content-Type: application/json`

#### Request Payload:
```json
{
  "Resin_Content": 48.0,
  "Pigment_Content": 24.0,
  "Solvent_Content": 20.0,
  "Additive_Content": 2.5,
  "Curing_Agent_Content": 5.5,
  "Curing_Temperature": 105.0,
  "Curing_Time": 45.0
}
```

#### Response (HTTP 200 OK):
```json
{
  "success": true,
  "predictions": {
    "Hardness": 77.32,
    "Adhesion": 76.38,
    "Flexibility": 43.74,
    "Gloss": 79.49,
    "Drying_Time": 57.06
  },
  "units": {
    "Hardness": "Index (0-100)",
    "Adhesion": "Index (0-100)",
    "Flexibility": "Index (0-100)",
    "Gloss": "GU @ 60°",
    "Drying_Time": "minutes"
  },
  "input_summary": { ... },
  "formulation_sum": 100.0,
  "is_demo": true
}
```

#### Error Response (HTTP 400 Bad Request):
```json
{
  "success": false,
  "error": "Pigment Content exceeds physical maximum of 100.0."
}
```

---

### Endpoint: `GET /health`
Returns backend and model status.

#### Response (HTTP 200 OK):
```json
{
  "status": "running",
  "model_loaded": true,
  "is_demo": true,
  "model_name": "Gradient Boosting Regressor (Multi-Output)",
  "features": ["Resin_Content", "Pigment_Content", "Solvent_Content", "Additive_Content", "Curing_Agent_Content", "Curing_Temperature", "Curing_Time"],
  "targets": ["Hardness", "Adhesion", "Flexibility", "Gloss", "Drying_Time"]
}
```

---

## 8. How to Replace the Demo Model with Real Experimental Model

If you have trained a new model on physical laboratory measurements:

1. Save your trained Scikit-learn estimator or Pipeline object to:
   ```text
   AI_Coating_Web/models/coating_model.joblib
   ```
2. If your model uses an independent feature scaler (e.g. `StandardScaler`), save it to:
   ```text
   AI_Coating_Web/models/preprocessing_pipeline.joblib
   ```
3. Update `models/model_metadata.json` with your experimental metrics:
   ```json
   {
     "model_name": "My Experimental Model",
     "is_demo": false,
     "overall_metrics": {
       "Mean_R2": 0.945,
       "Mean_MAE": 1.42,
       "Mean_RMSE": 1.85
     }
   }
   ```
4. Restart Flask:
   ```powershell
   python app.py
   ```
   The web application will instantly load your real experimental model and the demo banner will automatically disappear!

---

## 9. Troubleshooting & FAQ

| Problem | Cause | Solution |
| :--- | :--- | :--- |
| **Top badge shows "Model File Missing"** | `models/coating_model.joblib` was not found. | Run `python generate_demo_model.py` to regenerate model assets. |
| **`Port 5000 is already in use`** | Another application or previous Flask session is using port 5000. | Start Flask on another port: `$env:PORT=5001; python app.py`. |
| **ModuleNotFoundError: No module named 'flask'** | Virtual environment is not activated. | Run `.\venv\Scripts\activate` before launching. |
| **Prediction Error in Browser** | Non-numeric or missing input values. | Ensure all 7 input boxes contain valid numeric values. |
