"""
Ceylon Tea Production Forecasting - Ensemble Learning Dashboard (local, Streamlit)

Run:   streamlit run app.py        (opens http://localhost:8501)

Loads the models trained in Google Colab from the folder `tea_model_bundle/`
(next to this file). No training happens here; if the bundle cannot be loaded,
the app shows an error and stops.

Models (10, from 7 families): Seasonal naive (baseline) · Random Forest, Extra Trees (bagging) ·
Gradient Boosting, LightGBM, CatBoost (boosting) · ANN (neural network) · Stacking ·
SARIMA (time series) · Hybrid SARIMA + RF.

AI-use declaration: this code was generated with the help of Claude (Anthropic),
a generative AI tool, and reviewed and run by me.
Data: Sri Lanka Tea Board - Revised Tea Production reports, 2021–2025.
"""
import os
import json
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Ceylon Tea Forecasting", page_icon="🍃", layout="wide")

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(HERE, "srilanka_tea_production_monthly_2021_2025.csv")
BUNDLE = os.path.join(HERE, "tea_model_bundle")
ELEVS = ["High", "Medium", "Low"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
FEATURES = ["lag1", "lag2", "lag3", "lag12", "roll3", "month_sin", "month_cos", "t",
            "elev_High", "elev_Medium", "elev_Low"]
NAIVE, STACK = "Seasonal Naive (lag-12)", "Stacking (RF+GB+LGBM+Cat→Ridge)"
FAMILY = {NAIVE: "Baseline", "Random Forest": "Bagging", "Extra Trees": "Bagging",
          "Gradient Boosting": "Boosting", "LightGBM": "Boosting", "CatBoost": "Boosting",
          "ANN (MLP)": "Neural network", STACK: "Stacking", "SARIMA": "Time series", "Hybrid SARIMA + RF": "Hybrid"}
FAM_COL = {"Baseline": "#9E9E9E", "Bagging": "#2E7D32", "Boosting": "#1565C0", "Neural network": "#8E24AA",
           "Stacking": "#EF6C00", "Time series": "#6D4C41", "Hybrid": "#C62828"}
ELEV_COL = {"High": "#2E7D32", "Medium": "#F9A825", "Low": "#6D4C41", "Total": "#212121"}


# --------------------------------------------------------------------------- helpers (same as the notebook)
def make_features(df):
    long = df.melt(id_vars=["Date", "Year", "Month"], value_vars=ELEVS, var_name="Elevation", value_name="Prod")
    long["Prod"] = long["Prod"] / 1e6
    long = long.sort_values(["Elevation", "Date"]).reset_index(drop=True)
    g = long.groupby("Elevation")["Prod"]
    for L in [1, 2, 3, 12]:
        long[f"lag{L}"] = g.shift(L)
    long["roll3"] = g.transform(lambda s: s.shift(1).rolling(3).mean())
    long["month_sin"] = np.sin(2 * np.pi * long.Month / 12)
    long["month_cos"] = np.cos(2 * np.pi * long.Month / 12)
    long["t"] = (long.Year - 2021) * 12 + long.Month - 1
    for e in ELEVS:
        long[f"elev_{e}"] = (long.Elevation == e).astype(int)
    return long


def make_row(elev, month, lag1, lag2, lag3, lag12, t=60):
    return pd.DataFrame([{"lag1": lag1, "lag2": lag2, "lag3": lag3, "lag12": lag12,
                          "roll3": (lag1 + lag2 + lag3) / 3,
                          "month_sin": np.sin(2 * np.pi * month / 12), "month_cos": np.cos(2 * np.pi * month / 12),
                          "t": t, **{f"elev_{k}": int(k == elev) for k in ELEVS}}])[FEATURES]


def metrics(y, p):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    y, p = np.asarray(y), np.asarray(p)
    return {"RMSE": np.sqrt(mean_squared_error(y, p)), "MAE": mean_absolute_error(y, p),
            "MAPE %": np.mean(np.abs((y - p) / y)) * 100, "R²": r2_score(y, p)}


@st.cache_data
def load_data():
    return pd.read_csv(CSV_PATH, parse_dates=["Date"])


@st.cache_resource(show_spinner="Loading the trained models ...")
def load_models():
    try:
        import joblib
        final = joblib.load(os.path.join(BUNDLE, "ml_models.joblib"))
        results = pd.read_csv(os.path.join(BUNDLE, "results.csv"), index_col=0)
        by_elev = pd.read_csv(os.path.join(BUNDLE, "results_by_elevation.csv"))
        pred = pd.read_csv(os.path.join(BUNDLE, "predictions_2025.csv"), parse_dates=["Date"])
        imp = pd.read_csv(os.path.join(BUNDLE, "feature_importance.csv"), index_col=0)
        fc = pd.read_csv(os.path.join(BUNDLE, "forecast_2026_01.csv"), index_col=0)
        meta = json.load(open(os.path.join(BUNDLE, "meta.json")))
        test_row = make_row("Low", 1, 13, 13, 13, 13)
        for m in final.values():  # check every model works on this computer
            m.predict(test_row)
        source = f"✅ Using models trained in Google Colab ({meta.get('created', '')})"
    except Exception as e:
        st.error(f"Could not load the Colab model bundle from `{BUNDLE}`: {type(e).__name__}: {e}\n\n"
                 "Check that the `tea_model_bundle/` folder is complete and that the library versions "
                 "match requirements.txt.")
        st.stop()
    return dict(final=final, results=results, by_elev=by_elev, pred=pred, imp=imp, fc=fc, meta=meta, source=source)


# --------------------------------------------------------------------------- load
df = load_data()
M = load_models()
final, results, by_elev, pred, imp = M["final"], M["results"], M["by_elev"], M["pred"], M["imp"]
hist = make_features(df).set_index(["Elevation", "Date"])["Prod"]
ALL = results.index.tolist()
ML = [m for m in ALL if m in final]

fc = M["fc"]

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("🍃 About")
    st.write("Forecasting Sri Lanka's monthly tea production with ensemble learning.")
    for fam in FAM_COL:
        members = [m for m in ALL if FAMILY.get(m) == fam]
        if members:
            st.markdown(f"**{fam}:** {', '.join(members)}")
    st.divider()
    st.markdown(M["source"])
    if M["meta"]:
        import sklearn
        v = M["meta"].get("versions", {})
        st.caption(f"Trained with scikit-learn {v.get('scikit-learn')} · this computer: scikit-learn {sklearn.__version__}")
    st.divider()
    st.caption("Data: Sri Lanka Tea Board, Revised Tea Production reports, Jan 2021 – Dec 2025.")

# --------------------------------------------------------------------------- header
st.title("Ceylon Tea Production Forecasting")
st.caption("Bagging · Boosting · Stacking · Neural network · SARIMA · Hybrid - trained on 2022–2024, tested on 2025")
best = ALL[0]
best_total = fc.loc[best, "Total"] if best in fc.index else fc.loc[ML[0], "Total"]
k1, k2, k3, k4 = st.columns(4)
k1.metric("2025 production", f"{df[df.Year == 2025].Total.sum() / 1e6:.1f} Mn kg")
k2.metric("2022 vs 2021", f"{(df[df.Year == 2022].Total.sum() / df[df.Year == 2021].Total.sum() - 1) * 100:.1f}%")
k3.metric("Best model (2025 RMSE)", best, f"{results.RMSE.iloc[0]:.3f} Mn kg", delta_color="off")
k4.metric(f"Jan 2026 forecast ({best if best in fc.index else ML[0]})", f"{best_total:.1f} Mn kg")

tabs = st.tabs(["📊 Data overview", "🏆 Model comparison", "📈 Actual vs predicted",
                "🔍 Feature importance", "🧪 Try a prediction", "🔮 Jan 2026 forecast"])

# --------------------------------------------------------------------------- 1. data overview
with tabs[0]:
    view = st.radio("Series", ["All", "High", "Medium", "Low", "Total"], horizontal=True)
    f = go.Figure()
    for c in (["High", "Medium", "Low", "Total"] if view == "All" else [view]):
        f.add_scatter(x=df.Date, y=df[c] / 1e6, mode="lines+markers", name=c,
                      line=dict(color=ELEV_COL[c], width=3 if c == "Total" else 2))
    f.add_vrect(x0="2022-01-01", x1="2022-12-31", fillcolor="red", opacity=0.07, line_width=0,
                annotation_text="2022 crisis")
    f.update_layout(title="Monthly tea production (Mn kg)", template="plotly_white", height=420, hovermode="x unified")
    st.plotly_chart(f)
    c1, c2 = st.columns(2)
    y = (df.groupby("Year")[ELEVS].sum() / 1e6).reset_index().melt("Year", var_name="Elevation", value_name="Mn kg")
    fy = px.bar(y, x="Year", y="Mn kg", color="Elevation", color_discrete_map=ELEV_COL, title="Annual production by elevation")
    fy.update_layout(template="plotly_white", height=380)
    c1.plotly_chart(fy)
    hm = df.pivot(index="Year", columns="Month", values="Total") / 1e6
    fh = px.imshow(hm, text_auto=".1f", color_continuous_scale="Greens", aspect="auto",
                   labels=dict(x="Month", y="Year", color="Mn kg"), title="Seasonality heatmap - total production")
    fh.update_xaxes(tickvals=list(range(1, 13)), ticktext=MONTHS)
    fh.update_layout(height=380)
    c2.plotly_chart(fh)

# --------------------------------------------------------------------------- 2. comparison
with tabs[1]:
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
                  text=t[metric].round(3), title=f"{metric} on the 2025 test year - {elev}")
    fcmp.update_layout(template="plotly_white", height=520,
                       yaxis=dict(categoryorder="array", categoryarray=t.Model.tolist()[::-1]))
    st.plotly_chart(fcmp)
    st.dataframe(t.set_index("Model").round(3))
    st.caption("RMSE/MAE in million kg (lower is better). MAPE = average % error. "
               "R² over all elevations looks high because the elevations differ greatly in size.")

# --------------------------------------------------------------------------- 3. actual vs predicted
with tabs[2]:
    c1, c2 = st.columns([1, 3])
    el2 = c1.selectbox("Elevation", ["Low", "Medium", "High", "Total"])
    chosen = c2.multiselect("Models", ALL, default=ALL[:3] + ([NAIVE] if NAIVE not in ALL[:3] else []))
    gg = pred.groupby("Date").sum(numeric_only=True).reset_index() if el2 == "Total" else pred[pred.Elevation == el2]
    f = go.Figure()
    f.add_scatter(x=gg.Date, y=gg.Actual, name="Actual", mode="lines+markers", line=dict(color="black", width=4))
    for m in chosen:
        f.add_scatter(x=gg.Date, y=gg[m], name=m, mode="lines+markers", line=dict(dash="dash"))
    f.update_layout(title=f"{el2} grown - actual vs predicted, 2025", yaxis_title="Mn kg",
                    template="plotly_white", height=460, hovermode="x unified")
    st.plotly_chart(f)
    if chosen:
        st.dataframe(pd.DataFrame({m: metrics(gg.Actual, gg[m]) for m in chosen}).T.round(3))

# --------------------------------------------------------------------------- 4. feature importance
with tabs[3]:
    sel = st.multiselect("Models", list(imp.columns), default=["Random Forest", "CatBoost"])
    if sel:
        im = imp[sel].reset_index().rename(columns={"index": "Feature"}).melt("Feature", var_name="Model", value_name="Importance")
        order = imp[sel].mean(axis=1).sort_values().index.tolist()
        fi = px.bar(im, x="Importance", y="Feature", color="Model", barmode="group", orientation="h",
                    title="Permutation importance on 2025 (RMSE increase when a feature is shuffled)")
        fi.update_layout(template="plotly_white", height=480, yaxis=dict(categoryorder="array", categoryarray=order))
        st.plotly_chart(fi)
    st.markdown("- `lag1` = last month (momentum) · `lag12` = same month last year (seasonality) · "
                "`lag2`, `lag3` = 2–3 months ago · `roll3` = 3-month average\n"
                "- A larger bar means the model relies more on that input.")

# --------------------------------------------------------------------------- 5. try a prediction
with tabs[4]:
    st.write("Enter recent production figures (million kg) and every machine-learning model predicts the next month. "
             "The boxes start with the latest real values (Oct–Dec 2025). "
             "SARIMA and the hybrid need the full monthly history, so they are shown in the forecast tab instead.")
    c1, c2 = st.columns(2)
    p_el = c1.selectbox("Elevation", ELEVS, index=2, key="p_el")
    p_mo = c2.selectbox("Month to predict", MONTHS, index=0, key="p_mo")
    mo = MONTHS.index(p_mo) + 1
    d = (hist[p_el]["2025-12-01"], hist[p_el]["2025-11-01"], hist[p_el]["2025-10-01"], hist[p_el][f"2025-{mo:02d}-01"])
    k = f"{p_el}_{p_mo}"
    n1, n2, n3, n4 = st.columns(4)
    lag1 = n1.number_input("Last month", value=round(float(d[0]), 3), step=0.1, format="%.3f", key="l1" + k)
    lag2 = n2.number_input("2 months ago", value=round(float(d[1]), 3), step=0.1, format="%.3f", key="l2" + k)
    lag3 = n3.number_input("3 months ago", value=round(float(d[2]), 3), step=0.1, format="%.3f", key="l3" + k)
    lag12 = n4.number_input("Same month last year", value=round(float(d[3]), 3), step=0.1, format="%.3f", key="l12" + k)
    if min(lag1, lag2, lag3, lag12) <= 0:
        st.error("All values must be greater than 0.")
    else:
        X = make_row(p_el, mo, lag1, lag2, lag3, lag12)
        P = {n: float(final[n].predict(X)[0]) for n in ML}
        P[NAIVE] = float(lag12)
        tp = pd.DataFrame({"Model": list(P), "Predicted (Mn kg)": list(P.values())})
        tp["Family"] = tp.Model.map(FAMILY)
        tp["vs same month last year %"] = ((tp["Predicted (Mn kg)"] / lag12 - 1) * 100).round(1)
        tp = tp.sort_values("Predicted (Mn kg)")
        fp = px.bar(tp, x="Predicted (Mn kg)", y="Model", orientation="h", color="Family", color_discrete_map=FAM_COL,
                    text=tp["Predicted (Mn kg)"].round(3), title=f"Predicted {p_el} grown production for {p_mo}")
        fp.add_vline(x=lag12, line_dash="dot", annotation_text="same month last year")
        fp.update_layout(template="plotly_white", height=440)
        st.plotly_chart(fp)
        st.dataframe(tp.set_index("Model").round(3))
        st.caption("Predictions update instantly when you change a value. Try lowering 'Last month' to simulate a drought.")

# --------------------------------------------------------------------------- 6. forecast
with tabs[5]:
    st.write("Forecast for **January 2026** from the final models (trained on all data to Dec 2025).")
    order = [m for m in ALL if m in fc.index]
    sel = st.multiselect("Models ", order, default=order[:5])
    if sel:
        fm = fc.loc[sel, ELEVS].reset_index().rename(columns={"index": "Model"}).melt("Model", var_name="Elevation", value_name="Mn kg")
        ff = px.bar(fm, x="Model", y="Mn kg", color="Elevation", color_discrete_map=ELEV_COL,
                    title="January 2026 forecast by elevation (Mn kg)")
        ff.update_layout(template="plotly_white", height=440, xaxis_tickangle=-25)
        st.plotly_chart(ff)
    st.dataframe(fc.loc[order].round(3))
