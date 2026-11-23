# Weather-Driven Forecasting of Ceylon Tea Production — Streamlit dashboard

## Folder layout
```
tea_dashboard/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── srilanka_tea_production_monthly_2015_2025.csv   tea production dataset (also inside the bundle)
└── tea_model_bundle/          ← unzip tea_model_bundle.zip (downloaded from Colab) here
    ├── ml_models.joblib           7 trained ML models
    ├── results.csv                scores of all 10 models (test 2024–2025)
    ├── results_by_elevation.csv
    ├── predictions_test.csv
    ├── feature_importance.csv
    ├── weather_effect.csv         error with vs without weather features
    ├── forecast_2026_01.csv
    ├── tea_production_monthly.csv 2015–2025 production by elevation
    ├── tea_weather_monthly.csv    merged model table (production + weather features)
    ├── weather_daily.csv          ~33,000 daily weather records, 8 locations in the tea-growing regions
    └── meta.json
```

## Run (Mac Terminal, inside tea_dashboard)
```
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```
Opens **http://localhost:8501**. Stop with `Ctrl + C`. (LightGBM on Mac may need `brew install libomp`.)

If the app shows a red error, check that `tea_model_bundle/` is next to `app.py` (with `ml_models.joblib`
directly inside it) and that the library versions match requirements.txt.

## Data
- Tea production: Sri Lanka Tea Board figures via Forbes & Walker annual production summaries, Jan 2015 – Dec 2025.
- Weather: Open-Meteo historical archive (ERA5) at 8 tea towns chosen to represent Sri Lanka's official tea-growing regions:
  Nuwara Eliya (Nuwara Eliya), Hatton (Dimbula), Bandarawela (Uva) — high grown; Kandy, Matale (Kandy) — mid grown;
  Ratnapura (Sabaragamuwa), Galle and Deniyaya (Ruhuna) — low grown.

*AI-use declaration: code generated with help from Claude (Anthropic) and reviewed by me.*
