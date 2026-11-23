# Ceylon Tea Production Forecasting - local Streamlit dashboard

## Folder layout
```
tea-dashboard/
├── app.py
├── requirements.txt
├── srilanka_tea_production_monthly_2021_2025.csv
└── tea_model_bundle/          ← unzip tea_model_bundle.zip (downloaded from Colab) here
    ├── ml_models.joblib
    ├── results.csv
    ├── results_by_elevation.csv
    ├── predictions_2025.csv
    ├── feature_importance.csv
    ├── forecast_2026_01.csv
    └── meta.json
```

## Run (Mac Terminal, inside the tea-dashboard folder)
```
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```
The browser opens **http://localhost:8501**. Stop with `Ctrl + C`.

- The sidebar shows **"✅ Using models trained in Google Colab"** when the bundle loads.
- If the bundle cannot be loaded (missing folder or different library versions), the app shows an
  error and stops. It does not train models itself - re-run the Colab notebook to regenerate the bundle.
- LightGBM on Mac may need: `brew install libomp`

## Models (10, from 7 families)
Seasonal naive (baseline) · Random Forest, Extra Trees (bagging) · Gradient Boosting, LightGBM, CatBoost (boosting) ·
ANN/MLP (neural network) · Stacking (RF+GB+LightGBM+CatBoost → Ridge) · SARIMA (time series) · Hybrid SARIMA + RF

*AI-use declaration: code generated with help from Claude (Anthropic) and reviewed by me.*
