"""
Retail Sales Analytics Dashboard
Streamlit app that visualizes the EDA, ABC/Pareto analysis, RFM customer
segmentation, and sales forecasts produced in the project notebooks.

Run locally with:  streamlit run app.py   (from inside the dashboard/ folder)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from data_utils import (
    abc_classification,
    abc_summary,
    compute_kpis,
    filter_by_date,
    load_raw_data,
    monthly_trend,
    moving_average_forecast,
    revenue_by_dow,
    revenue_by_month_name,
    revenue_volume_quadrant,
    rfm_segmentation,
    top_products,
    weekly_trend,
)

st.set_page_config(
    page_title="Retail Sales Analytics Dashboard",
    page_icon="📊",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "retail_sales_cleaned.csv"
MODELS_DIR = BASE_DIR / "models"


@st.cache_data
def get_raw_data():
    return load_raw_data(str(DATA_PATH))


@st.cache_data
def get_abc(df):
    return abc_classification(df)


@st.cache_data
def get_rfm(df):
    return rfm_segmentation(df)


@st.cache_data
def get_precomputed_forecast(filename):
    """Prefer the forecast CSVs saved by 04_sales_forecasting.ipynb (fast, no
    retraining needed). Falls back to None if the notebook hasn't been run yet."""
    path = MODELS_DIR / filename
    if path.exists():
        fdf = pd.read_csv(path)
        fdf["date"] = pd.to_datetime(fdf["date"])
        return fdf
    return None


df_raw = get_raw_data()

st.sidebar.title("📊 Retail Sales Analytics")
st.sidebar.caption("End-to-end retail/FMCG data project")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Sales Trends", "Products & ABC Analysis", "Customers & RFM", "Forecasting", "About"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Filter data")
min_date, max_date = df_raw["date"].min().date(), df_raw["date"].max().date()
date_range = st.sidebar.date_input(
    "Date range", value=(min_date, max_date), min_value=min_date, max_value=max_date
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

df = filter_by_date(df_raw, start_date, end_date)

st.sidebar.markdown("---")
st.sidebar.caption(f"{len(df):,} transaction rows in selected range")
st.sidebar.caption("Data: retail_sales_cleaned.csv")

if page == "Overview":
    st.title("Retail Sales Overview")
    st.caption(f"{start_date} to {end_date}")

    kpis = compute_kpis(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Revenue", f"R{kpis['total_revenue']:,.0f}")
    c2.metric("Total Volume", f"{kpis['total_kg']:,.0f} kg")
    c3.metric("Invoices", f"{kpis['n_invoices']:,}")
    c4.metric("Avg Transaction Value", f"R{kpis['avg_txn_value']:,.0f}")

    c5, c6 = st.columns(2)
    c5.metric("Active Products", f"{kpis['n_products']:,}")
    c6.metric("Active Customers", f"{kpis['n_customers']:,}")

    st.markdown("### Monthly Revenue Trend")
    mt = monthly_trend(df)
    fig = px.line(mt, x="year_month", y="sales_value", markers=True)
    fig.update_layout(xaxis_title="Month", yaxis_title="Revenue", height=400)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Revenue by Day of Week")
        dow = revenue_by_dow(df)
        fig = px.bar(dow, x="day_name", y="sales_value", color="sales_value", color_continuous_scale="Blues")
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.markdown("### Revenue by Calendar Month")
        mn = revenue_by_month_name(df)
        fig = px.bar(mn, x="month_name", y="sales_value", color="sales_value", color_continuous_scale="Greens")
        fig.update_layout(showlegend=False, coloraxis_showscale=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

elif page == "Sales Trends":
    st.title("Sales Trends")

    st.markdown("### Daily Revenue")
    daily = df.groupby("date")["sales_value"].sum().reset_index()
    fig = px.line(daily, x="date", y="sales_value")
    fig.update_traces(line_width=1)
    fig.update_layout(height=350, yaxis_title="Revenue")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Weekly Revenue")
    wt = weekly_trend(df)
    fig = px.line(wt, x="date", y="sales_value", markers=True)
    fig.update_layout(height=350, yaxis_title="Revenue")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Growth: First 6 Months vs Most Recent 6 Months")
    mt = monthly_trend(df).set_index("year_month")["sales_value"]
    if len(mt) >= 8:
        first_6 = mt.iloc[1:7].mean()
        last_6 = mt.iloc[-7:-1].mean()
        growth = (last_6 - first_6) / first_6 * 100 if first_6 else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Avg — First 6 Months", f"R{first_6:,.0f}")
        c2.metric("Avg — Most Recent 6 Months", f"R{last_6:,.0f}")
        c3.metric("Growth", f"{growth:+.1f}%")
    else:
        st.info("Select a wider date range to compute growth over 6-month windows.")

elif page == "Products & ABC Analysis":
    st.title("Product Analytics & ABC / Pareto Classification")

    n_top = st.slider("Show top N products", 5, 50, 20)
    metric = st.selectbox("Rank by", ["revenue", "units", "kg"])
    tp = top_products(df, n=n_top, by=metric)
    st.markdown(f"### Top {n_top} Products by {metric.title()}")
    fig = px.bar(tp.sort_values(metric), x=metric, y="product", orientation="h", height=max(400, n_top * 22))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("## ABC / Pareto Analysis")
    st.caption(
        "Products ranked by revenue and split into A (top ~80% of cumulative revenue), "
        "B (next ~15%), and C (remaining ~5%)."
    )

    product_abc = get_abc(df)
    summary = abc_summary(product_abc)

    c1, c2, c3 = st.columns(3)
    for col, (_, row) in zip([c1, c2, c3], summary.iterrows()):
        col.metric(
            f"Class {row['abc_class']}",
            f"{row['n_products']:.0f} products",
            f"{row['pct_of_revenue']:.1f}% of revenue",
        )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(summary, names="abc_class", values="n_products", title="Share of Catalogue",
                     color="abc_class", color_discrete_map={"A": "#2ca02c", "B": "#ff9900", "C": "#d62728"})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.pie(summary, names="abc_class", values="total_revenue", title="Share of Revenue",
                     color="abc_class", color_discrete_map={"A": "#2ca02c", "B": "#ff9900", "C": "#d62728"})
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Revenue vs Volume Quadrant")
    quad = revenue_volume_quadrant(product_abc)
    fig = px.scatter(
        quad, x="kg", y="revenue", color="quadrant", hover_name="product",
        log_x=True, log_y=True, opacity=0.6,
        color_discrete_map={
            "High Revenue / High Volume": "#2ca02c",
            "High Revenue / Low Volume (Premium)": "#1f77b4",
            "Low Revenue / High Volume (Commodity)": "#ff7f0e",
            "Low Revenue / Low Volume": "#d62728",
        },
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View Class A products (highest priority)"):
        st.dataframe(
            product_abc[product_abc["abc_class"] == "A"][["product", "revenue", "units", "kg"]]
            .sort_values("revenue", ascending=False),
            use_container_width=True,
        )

elif page == "Customers & RFM":
    st.title("Customer Analytics & RFM Segmentation")
    st.caption("Recency, Frequency, Monetary scoring mapped to 8 standard customer segments.")

    rfm = get_rfm(df)
    segment_counts = rfm["segment"].value_counts().reset_index()
    segment_counts.columns = ["segment", "count"]
    segment_value = rfm.groupby("segment")["monetary"].sum().reset_index()

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(segment_counts.sort_values("count"), x="count", y="segment", orientation="h",
                     title="Customers per Segment")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(segment_value.sort_values("monetary"), x="monetary", y="segment", orientation="h",
                     title="Revenue per Segment")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Frequency vs Monetary (bubble size = recency, bigger = more recent)")
    rfm_plot = rfm.copy()
    rfm_plot["bubble_size"] = rfm_plot["recency"].apply(lambda x: max(8, 60 - x / 10))
    fig = px.scatter(
        rfm_plot, x="frequency", y="monetary", color="segment", size="bubble_size",
        hover_name="customer", log_y=True,
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Segment Explorer")
    selected_segment = st.selectbox("Choose a segment", sorted(rfm["segment"].unique()))
    seg_df = rfm[rfm["segment"] == selected_segment].sort_values("monetary", ascending=False)
    st.caption(seg_df["recommended_action"].iloc[0] if len(seg_df) else "")
    st.dataframe(
        seg_df[["customer", "recency", "frequency", "monetary", "segment"]],
        use_container_width=True,
    )

elif page == "Forecasting":
    st.title("Sales Forecasting")

    horizon_map = {
        "30 Days": "forecast_30_day.csv",
        "6 Months": "forecast_6_month.csv",
        "12 Months": "forecast_12_month.csv",
    }
    horizon_label = st.radio("Forecast horizon", list(horizon_map.keys()), horizontal=True)
    fdf = get_precomputed_forecast(horizon_map[horizon_label])

    used_fallback = False
    if fdf is None:
        st.warning(
            "No saved forecast found in `models/`. Showing a lightweight fallback "
            "(7-day moving average with day-of-week seasonality) instead. "
            "Run `04_sales_forecasting.ipynb` to generate the full trained-model forecast."
        )
        horizon_days = {"30 Days": 30, "6 Months": 182, "12 Months": 365}[horizon_label]
        fdf = moving_average_forecast(df_raw, horizon_days=horizon_days)
        used_fallback = True

    c1, c2 = st.columns(2)
    c1.metric(f"Forecasted Total ({horizon_label})", f"R{fdf['forecast_sales'].sum():,.0f}")
    c2.metric("Forecasted Avg / Day", f"R{fdf['forecast_sales'].mean():,.0f}")

    recent = df_raw[df_raw["date"] >= df_raw["date"].max() - pd.Timedelta(days=90)]
    recent_daily = recent.groupby("date")["sales_value"].sum().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recent_daily["date"], y=recent_daily["sales_value"],
                              name="Actual (last 90 days)", line=dict(color="steelblue")))
    fig.add_trace(go.Scatter(x=fdf["date"], y=fdf["forecast_sales"],
                              name=f"Forecast ({horizon_label})", line=dict(color="crimson", dash="dash")))
    fig.update_layout(height=450, yaxis_title="Revenue", xaxis_title="Date")
    st.plotly_chart(fig, use_container_width=True)

    if horizon_label != "30 Days" and not used_fallback:
        st.caption(
            "⚠️ Longer-horizon forecasts are recursive (each prediction feeds the next) and the model has "
            "under two years of history — treat 6/12-month figures as directional trend, not precise daily numbers."
        )

    comparison_path = MODELS_DIR / "model_comparison.csv"
    if comparison_path.exists():
        st.markdown("### Model Comparison (from 04_sales_forecasting.ipynb)")
        comp = pd.read_csv(comparison_path, index_col=0)
        st.dataframe(comp.round(2), use_container_width=True)

else:
    st.title("About This Project")
    st.markdown(
        """
This dashboard is the interactive layer on top of a 4-notebook, end-to-end retail/FMCG
analytics project:

1. **EDA & Business Questions** — sales performance, product analytics, customer analytics
2. **ABC / Pareto Analysis** — product classification for inventory prioritization
3. **RFM Customer Segmentation** — 8 actionable customer segments
4. **Sales Forecasting** — baselines, statistical models, and ML models compared on MAE/RMSE/MAPE/WAPE,
   with 30-day / 6-month / 12-month forward forecasts

**Data:** 181,670 invoice-line transactions, 1,247 products, 50 customers, Sep 2024 – Jul 2026.

**Stack:** Python, pandas, scikit-learn, Plotly, Streamlit.

See the full notebooks and README on [GitHub](#) for the complete analysis and methodology.
        """
    )