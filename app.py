"""
Weather-Driven Forecasting of Ceylon Tea Production — Ensemble Learning Dashboard (local, Streamlit)

Run:   python3 -m streamlit run app.py        (opens http://localhost:8501)

Everything is loaded from the folder `tea_model_bundle/` (next to this file), exported by the Colab notebook:
trained models, results, the tea production data and the daily weather of 8 tea-region locations.
No training happens here; if the bundle cannot be loaded, the app shows an error and stops.

Models (10, from 7 families): Seasonal naive (baseline) · Random Forest, Extra Trees (bagging) ·
Gradient Boosting, LightGBM, CatBoost (boosting) · ANN (neural network) · Stacking ·
SARIMA (time series) · Hybrid SARIMA + RF.

AI-use declaration: this code was generated with the help of Claude (Anthropic), a generative AI tool,
and reviewed and run by me.
Data: Sri Lanka Tea Board production figures (via Forbes & Walker), 2015–2025;
daily weather at 8 locations representing Sri Lanka's official tea-growing regions,
from the Open-Meteo historical archive (ERA5).
"""
import os
import json
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Ceylon Tea × Weather Forecasting", page_icon="🍃", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.join(HERE, "tea_model_bundle")
ELEVS = ["High", "Medium", "Low"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
NAIVE = "Seasonal Naive (lag-12)"
FAM_COL = {"Baseline": "#9E9E9E", "Bagging": "#2E7D32", "Boosting": "#1565C0", "Neural network": "#8E24AA",
           "Stacking": "#EF6C00", "Time series": "#6D4C41", "Hybrid": "#C62828"}
ELEV_COL = {"High": "#2E7D32", "Medium": "#F9A825", "Low": "#6D4C41", "Total": "#212121", "Green": "#8BC34A"}
CLIM_VARS = ["rain_mm", "rain_days", "rain_hours", "temp_mean", "temp_max", "temp_min", "sun_hours", "radiation", "et0"]


def metrics(y, p):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    y, p = np.asarray(y), np.asarray(p)
    return {"RMSE": np.sqrt(mean_squared_error(y, p)), "MAE": mean_absolute_error(y, p),
            "MAPE %": np.mean(np.abs((y - p) / y)) * 100, "R²": r2_score(y, p)}


def rd(name, **kw):
    return pd.read_csv(os.path.join(BUNDLE, name), **kw)


@st.cache_resource(show_spinner="Loading the trained models and data ...")
def load_bundle():
    try:
        import joblib
        B = dict(
            final=joblib.load(os.path.join(BUNDLE, "ml_models.joblib")),
            results=rd("results.csv", index_col=0),
            by_elev=rd("results_by_elevation.csv"),
            pred=rd("predictions_test.csv", parse_dates=["Date"]),
            imp=rd("feature_importance.csv", index_col=0),
            weff=rd("weather_effect.csv", index_col=0),
            fc=rd("forecast_2026_01.csv", index_col=0),
            tea=rd("tea_production_monthly.csv", parse_dates=["Date"]),
            full=rd("tea_weather_monthly.csv", parse_dates=["Date"]),
            daily=rd("weather_daily.csv", parse_dates=["date"]),
            meta=json.load(open(os.path.join(BUNDLE, "meta.json"))),
        )
        feats = B["meta"]["features"]
        sample = B["full"].dropna(subset=feats).iloc[[0]][feats]
        for m in B["final"].values():  # check every model works on this computer
            m.predict(sample)
    except Exception as e:
        st.error(f"Could not load the Colab model bundle from `{BUNDLE}`: {type(e).__name__}: {e}\n\n"
                 "Check that the `tea_model_bundle/` folder (unzipped from Colab) is next to app.py "
                 "and that the library versions match requirements.txt.")
        st.stop()
    return B


B = load_bundle()
final, results, by_elev, pred, imp, weff, fc = (B[k] for k in ["final", "results", "by_elev", "pred", "imp", "weff", "fc"])
tea, full, daily, meta = B["tea"], B["full"], B["daily"], B["meta"]
FEATURES, FAMILY, LOCATIONS = meta["features"], meta["family"], meta["locations"]
ALL = results.index.tolist()
ML = [m for m in ALL if m in final]

# climatology (normal weather) per elevation & month, from the training years — used for the scenario tab
clim = full[full.Year.between(2015, 2023)].groupby(["Elevation", "Month"])[CLIM_VARS].mean()
prod = full.dropna(subset=["Prod"]).set_index(["Elevation", "Date"])["Prod"]

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("🍃 About")
    st.write("Predicting Sri Lanka's monthly tea production from recent production **and the weather in each tea-growing region**, "
             "comparing ensemble learning models.")
    for fam in FAM_COL:
        members = [m for m in ALL if FAMILY.get(m) == fam]
        if members:
            st.markdown(f"**{fam}:** {', '.join(members)}")
    st.divider()
    st.markdown(f"✅ Using models trained in Google Colab ({meta.get('created', '')})")
    import sklearn
    st.caption(f"Trained with scikit-learn {meta['versions'].get('scikit-learn')} · this computer: {sklearn.__version__}")
    st.divider()
    st.caption("Data: Sri Lanka Tea Board production (via Forbes & Walker), Jan 2015 – Dec 2025 · "
               f"{len(daily):,} daily weather records from {len(LOCATIONS)} locations in the tea-growing regions (Open-Meteo / ERA5).")

# --------------------------------------------------------------------------- header
st.title("Weather-Driven Forecasting of Ceylon Tea Production")
st.caption(f"Train 2016–2023 · test on unseen 2024–2025 · {len(FEATURES)} features · 10 models from 7 families")
best = ALL[0]
k1, k2, k3, k4 = st.columns(4)
k1.metric("Tea data", f"{len(tea)} months", "2015 – 2025", delta_color="off")
k2.metric("Weather records", f"{len(daily):,}", f"{len(LOCATIONS)} locations · {daily.region.nunique()} tea regions", delta_color="off")
k3.metric("Best model (test RMSE)", best, f"{results.RMSE.iloc[0]:.3f} Mn kg", delta_color="off")
k4.metric("Jan 2026 forecast (best model)", f"{fc.loc[best, 'Total']:.1f} Mn kg" if best in fc.index else "—")

tabs = st.tabs(["📊 Tea & weather", "🌦️ Tea-region weather", "🏆 Model comparison", "📈 Actual vs predicted",
                "🔍 Does weather matter?", "🧪 Try a prediction", "🔮 Jan 2026 forecast"])

# --------------------------------------------------------------------------- 1. tea & weather
with tabs[0]:
    view = st.radio("Series", ["All", "High", "Medium", "Low", "Total"], horizontal=True)
    f = go.Figure()
    for c in (["High", "Medium", "Low", "Total"] if view == "All" else [view]):
        f.add_scatter(x=tea.Date, y=tea[c] / 1e6, mode="lines", name=c,
                      line=dict(color=ELEV_COL[c], width=3 if c == "Total" else 2))
    for a, b_, lab in [("2016-12-01", "2017-03-01", "Drought"), ("2020-03-01", "2020-04-30", "COVID"),
                       ("2021-05-01", "2022-12-31", "Fertiliser crisis")]:
        f.add_vrect(x0=a, x1=b_, fillcolor="red", opacity=0.07, line_width=0, annotation_text=lab)
    f.update_layout(title="Monthly tea production (Mn kg)", template="plotly_white", height=400, hovermode="x unified")
    st.plotly_chart(f)

    c1, c2 = st.columns(2)
    y = (tea.groupby("Year")[ELEVS + ["Green"]].sum() / 1e6).reset_index().melt("Year", var_name="Elevation", value_name="Mn kg")
    fy = px.bar(y, x="Year", y="Mn kg", color="Elevation", color_discrete_map=ELEV_COL, title="Annual production")
    fy.update_layout(template="plotly_white", height=380)
    c1.plotly_chart(fy)
    el = c2.selectbox("Production vs rainfall for", ELEVS, index=0)
    d = full[(full.Elevation == el) & full.Prod.notna()]
    fr = make_subplots(specs=[[{"secondary_y": True}]])
    fr.add_bar(x=d.Date, y=d.rain_mm, name="Rainfall (mm)", marker_color="#90CAF9", secondary_y=False)
    fr.add_scatter(x=d.Date, y=d.Prod, name="Production (Mn kg)", line=dict(color=ELEV_COL[el], width=2), secondary_y=True)
    fr.update_layout(title=f"{el} grown: monthly rainfall at its tea locations vs production", template="plotly_white", height=380)
    c2.plotly_chart(fr)

# --------------------------------------------------------------------------- 2. tea-region weather
with tabs[1]:
    pts = pd.DataFrame([{"Location": k, "Elevation": v[0], "Tea region": v[1], "lat": v[2], "lon": v[3]} for k, v in LOCATIONS.items()])
    c1, c2 = st.columns([1, 2])
    fm = px.scatter_geo(pts, lat="lat", lon="lon", color="Elevation", text="Location", hover_data=["Tea region"],
                        color_discrete_map=ELEV_COL, title="Weather locations in the tea-growing regions")
    fm.update_traces(marker=dict(size=12), textposition="top center")
    fm.update_geos(fitbounds="locations", showcountries=True, showland=True, landcolor="#F1F8E9")
    fm.update_layout(height=460, margin=dict(l=0, r=0, t=40, b=0))
    c1.plotly_chart(fm)
    dist = c2.selectbox("Location", list(LOCATIONS), format_func=lambda k: f"{k} ({LOCATIONS[k][1]} region, {LOCATIONS[k][0]} grown)")
    var = c2.radio("Variable", ["precipitation_sum", "temperature_2m_mean", "sunshine_duration"], horizontal=True,
                   format_func={"precipitation_sum": "Rainfall (mm)", "temperature_2m_mean": "Mean temperature (°C)",
                                "sunshine_duration": "Sunshine (hours)"}.get)
    dd = daily[daily.location == dist].set_index("date")[var]
    if var == "sunshine_duration":
        dd = dd / 3600
    mo = dd.resample("MS").sum() if var != "temperature_2m_mean" else dd.resample("MS").mean()
    fd = px.line(mo.reset_index(), x="date", y=var, title=f"{dist}: monthly {'total' if var != 'temperature_2m_mean' else 'average'}")
    fd.update_layout(template="plotly_white", height=380, yaxis_title="")
    c2.plotly_chart(fd)
    c1.dataframe(pts[["Location", "Tea region", "Elevation"]].set_index("Location"))
    st.caption(f"{len(daily):,} daily records · {daily.date.min().date()} to {daily.date.max().date()} · "
               "one or two well-known tea towns chosen to represent each official tea-growing region "
               "(Uda Pussellawa is next to Nuwara Eliya, so it is covered by Nuwara Eliya).")

# --------------------------------------------------------------------------- 3. comparison
with tabs[2]:
    c1, c2 = st.columns(2)
    metric = c1.radio("Metric", ["RMSE", "MAE", "MAPE %", "R²"], horizontal=True)
    elev = c2.radio("Elevation", ["All", "High", "Medium", "Low"], horizontal=True)
    if elev == "All":
        t = results.reset_index().rename(columns={"index": "Model"})
    else:
        t = by_elev[by_elev.Elevation == elev].drop(columns="Elevation").copy()
        t.insert(1, "Family", t.Model.map(FAMILY))
    t = t.sort_values(metric, ascending=(metric != "R²"))
    fcmp = px.bar(t, x=metric, y="Model", orientation="h", color="Family", color_discrete_map=FAM_COL,
                  text=t[metric].round(3), title=f"{metric} on the unseen 2024–2025 test years — {elev}")
    fcmp.update_layout(template="plotly_white", height=520,
                       yaxis=dict(categoryorder="array", categoryarray=t.Model.tolist()[::-1]))
    st.plotly_chart(fcmp)
    st.dataframe(t.set_index("Model").round(3))
    st.caption("RMSE/MAE in million kg (lower is better). MAPE = average % error.")

# --------------------------------------------------------------------------- 4. actual vs predicted
with tabs[3]:
    c1, c2 = st.columns([1, 3])
    el2 = c1.selectbox("Elevation ", ["Low", "Medium", "High", "Total"])
    chosen = c2.multiselect("Models", ALL, default=ALL[:3] + ([NAIVE] if NAIVE not in ALL[:3] else []))
    gg = pred.groupby("Date").sum(numeric_only=True).reset_index() if el2 == "Total" else pred[pred.Elevation == el2]
    f = go.Figure()
    f.add_scatter(x=gg.Date, y=gg.Actual, name="Actual", mode="lines+markers", line=dict(color="black", width=4))
    for m in chosen:
        f.add_scatter(x=gg.Date, y=gg[m], name=m, mode="lines+markers", line=dict(dash="dash"))
    f.update_layout(title=f"{el2} grown — actual vs predicted, 2024–2025", yaxis_title="Mn kg",
                    template="plotly_white", height=460, hovermode="x unified")
    st.plotly_chart(f)
    if chosen:
        st.dataframe(pd.DataFrame({m: metrics(gg.Actual, gg[m]) for m in chosen}).T.round(3))

# --------------------------------------------------------------------------- 5. does weather matter?
with tabs[4]:
    c1, c2 = st.columns(2)
    w = weff.reset_index().melt("Model", value_vars=["RMSE with weather", "RMSE without weather"], var_name="Inputs", value_name="RMSE")
    fw = px.bar(w, x="Model", y="RMSE", color="Inputs", barmode="group", title="Test error with vs without weather features",
                color_discrete_map={"RMSE with weather": "#1565C0", "RMSE without weather": "#BDBDBD"})
    fw.update_layout(template="plotly_white", height=420)
    c1.plotly_chart(fw)
    c1.dataframe(weff.round(3))
    sel = c2.multiselect("Models", list(imp.columns), default=["Random Forest", "CatBoost"])
    if sel:
        top = imp[sel].mean(axis=1).sort_values().index[-15:]
        im = imp.loc[top, sel].reset_index().rename(columns={"index": "Feature"}).melt("Feature", var_name="Model", value_name="Importance")
        fi = px.bar(im, x="Importance", y="Feature", color="Model", barmode="group", orientation="h",
                    title="Top 15 features (permutation importance)")
        fi.update_layout(template="plotly_white", height=520, yaxis=dict(categoryorder="array", categoryarray=list(top)))
        c2.plotly_chart(fi)
    st.markdown("`lag1/lag2/lag3` = production 1–3 months ago · `lag12` = same month last year · `rain_mm` = rainfall this month · "
                "`rain_lag1/2` = rainfall 1–2 months ago · `rain_3m` = 3-month rainfall · `water_balance` = rain − evapotranspiration · "
                "`rain_anom` = rain vs normal · `sun_hours` = sunshine")

# --------------------------------------------------------------------------- 6. try a prediction
with tabs[5]:
    st.write("Set recent production and the weather for a month, and every machine-learning model predicts that month's "
             "production. Production boxes start with the latest real values (late 2025); weather starts at the **normal** "
             "for that month in that region. Try the rainfall slider to simulate a drought or a very wet month.")
    c1, c2, c3 = st.columns(3)
    p_el = c1.selectbox("Elevation", ELEVS, index=2, key="p_el")
    p_mo = c2.selectbox("Month to predict (2026)", MONTHS, index=0, key="p_mo")
    pct = c3.slider("Rainfall vs normal (%)", 0, 250, 100, step=10, help="100% = normal rainfall for that month and region")
    mo = MONTHS.index(p_mo) + 1
    prev = lambda k: (mo - 1 - k) % 12 + 1  # month number k months before
    cl = clim.loc[(p_el, mo)]
    k = f"{p_el}_{p_mo}"
    st.markdown("**Recent production (Mn kg)**")
    n1, n2, n3, n4 = st.columns(4)
    lag1 = n1.number_input("Last month", value=round(float(prod[(p_el, pd.Timestamp("2025-12-01"))]), 3), step=0.1, format="%.3f", key="l1" + k)
    lag2 = n2.number_input("2 months ago", value=round(float(prod[(p_el, pd.Timestamp("2025-11-01"))]), 3), step=0.1, format="%.3f", key="l2" + k)
    lag3 = n3.number_input("3 months ago", value=round(float(prod[(p_el, pd.Timestamp("2025-10-01"))]), 3), step=0.1, format="%.3f", key="l3" + k)
    lag12 = n4.number_input("Same month last year", value=round(float(prod[(p_el, pd.Timestamp(f"2025-{mo:02d}-01"))]), 3), step=0.1, format="%.3f", key="l12" + k)
    st.markdown(f"**Weather at the {p_el.lower()}-grown tea locations**")
    w1, w2, w3, w4, w5 = st.columns(5)
    rain = w1.number_input("Rainfall this month (mm)", value=round(float(cl.rain_mm * pct / 100), 1), step=10.0, key="r0" + k + str(pct))
    rain1 = w2.number_input("Rainfall last month (mm)", value=round(float(clim.loc[(p_el, prev(1))].rain_mm * pct / 100), 1), step=10.0, key="r1" + k + str(pct))
    rain2 = w3.number_input("Rainfall 2 months ago (mm)", value=round(float(clim.loc[(p_el, prev(2))].rain_mm * pct / 100), 1), step=10.0, key="r2" + k + str(pct))
    temp = w4.number_input("Mean temperature (°C)", value=round(float(cl.temp_mean), 1), step=0.5, key="t" + k)
    sun = w5.number_input("Sunshine (hours/month)", value=round(float(cl.sun_hours), 0), step=10.0, key="s" + k)

    if min(lag1, lag2, lag3, lag12) <= 0 or min(rain, rain1, rain2) < 0:
        st.error("Production must be greater than 0 and rainfall cannot be negative.")
    else:
        ratio = rain / cl.rain_mm if cl.rain_mm > 0 else 1.0
        row = {"lag1": lag1, "lag2": lag2, "lag3": lag3, "lag12": lag12, "roll3": (lag1 + lag2 + lag3) / 3,
               "rain_mm": rain, "rain_days": min(cl.rain_days * ratio, 31), "rain_hours": min(cl.rain_hours * ratio, 744),
               "temp_mean": temp, "temp_max": cl.temp_max + (temp - cl.temp_mean), "temp_min": cl.temp_min + (temp - cl.temp_mean),
               "sun_hours": sun, "radiation": cl.radiation, "et0": cl.et0,
               "rain_lag1": rain1, "rain_lag2": rain2, "rain_3m": rain + rain1 + rain2,
               "temp_lag1": clim.loc[(p_el, prev(1))].temp_mean, "sun_lag1": clim.loc[(p_el, prev(1))].sun_hours,
               "water_balance": rain - cl.et0, "rain_anom": rain - cl.rain_mm,
               "month_sin": np.sin(2 * np.pi * mo / 12), "month_cos": np.cos(2 * np.pi * mo / 12),
               "t": (2026 - 2015) * 12 + mo - 1, **{f"elev_{e}": int(e == p_el) for e in ELEVS}}
        X = pd.DataFrame([row])[FEATURES]
        P = {n: float(final[n].predict(X)[0]) for n in ML}
        P[NAIVE] = float(lag12)
        tp = pd.DataFrame({"Model": list(P), "Predicted (Mn kg)": list(P.values())})
        tp["Family"] = tp.Model.map(FAMILY)
        tp["vs same month last year %"] = ((tp["Predicted (Mn kg)"] / lag12 - 1) * 100).round(1)
        tp = tp.sort_values("Predicted (Mn kg)")
        fp = px.bar(tp, x="Predicted (Mn kg)", y="Model", orientation="h", color="Family", color_discrete_map=FAM_COL,
                    text=tp["Predicted (Mn kg)"].round(3), title=f"Predicted {p_el} grown production for {p_mo} 2026 "
                                                            f"(rainfall {rain / cl.rain_mm * 100 if cl.rain_mm else 0:.0f}% of normal)")
        fp.add_vline(x=lag12, line_dash="dot", annotation_text="same month last year")
        fp.update_layout(template="plotly_white", height=440)
        st.plotly_chart(fp)
        st.dataframe(tp.set_index("Model").round(3))
        st.caption("SARIMA and the hybrid need the full monthly history, so they appear in the forecast tab instead.")

# --------------------------------------------------------------------------- 7. forecast
with tabs[6]:
    st.write("Forecast for **January 2026** from the final models (trained on 2016–2025), using the **actual January 2026 weather** "
             "at each tea location.")
    order = [m for m in ALL if m in fc.index]
    sel = st.multiselect("Models ", order, default=order[:5])
    if sel:
        fmm = fc.loc[sel, ELEVS].reset_index().rename(columns={"index": "Model"}).melt("Model", var_name="Elevation", value_name="Mn kg")
        ff = px.bar(fmm, x="Model", y="Mn kg", color="Elevation", color_discrete_map=ELEV_COL,
                    title="January 2026 forecast by elevation (Mn kg)")
        ff.update_layout(template="plotly_white", height=440, xaxis_tickangle=-25)
        st.plotly_chart(ff)
    st.dataframe(fc.loc[order].round(3))
