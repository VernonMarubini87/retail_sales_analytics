"""
Pure data-processing functions for the retail sales dashboard.
Kept separate from app.py (which holds the Streamlit/Plotly UI code) so this
module has no Streamlit dependency and can be unit-tested with plain pandas.
"""
import numpy as np
import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["date"] = pd.to_datetime(df["date"])
    return df


def filter_by_date(df: pd.DataFrame, start_date, end_date) -> pd.DataFrame:
    mask = (df["date"] >= pd.Timestamp(start_date)) & (df["date"] <= pd.Timestamp(end_date))
    return df.loc[mask].copy()


def compute_kpis(df: pd.DataFrame) -> dict:
    total_revenue = df["sales_value"].sum()
    total_kg = df["mass_kg"].sum()
    n_invoices = df["doc_no"].nunique()
    avg_txn_value = df.groupby("doc_no")["sales_value"].sum().mean() if n_invoices else 0
    n_products = df["product"].nunique()
    n_customers = df["customer"].nunique()
    return {
        "total_revenue": total_revenue,
        "total_kg": total_kg,
        "n_invoices": n_invoices,
        "avg_txn_value": avg_txn_value,
        "n_products": n_products,
        "n_customers": n_customers,
    }


def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    m = df.groupby("year_month")["sales_value"].sum().reset_index()
    m["year_month"] = pd.to_datetime(m["year_month"])
    return m.sort_values("year_month")


def weekly_trend(df: pd.DataFrame) -> pd.DataFrame:
    w = df.set_index("date")["sales_value"].resample("W").sum().reset_index()
    return w


def revenue_by_dow(df: pd.DataFrame) -> pd.DataFrame:
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    r = df.groupby("day_name")["sales_value"].sum().reindex(dow_order).reset_index()
    r.columns = ["day_name", "sales_value"]
    return r


def revenue_by_month_name(df: pd.DataFrame) -> pd.DataFrame:
    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    r = df.groupby("month_name")["sales_value"].sum().reindex(month_order).reset_index()
    r.columns = ["month_name", "sales_value"]
    return r.dropna()


def top_products(df: pd.DataFrame, n: int = 20, by: str = "revenue") -> pd.DataFrame:
    agg = df.groupby("product").agg(
        revenue=("sales_value", "sum"),
        units=("qty", "sum"),
        kg=("mass_kg", "sum"),
        n_invoices=("doc_no", "nunique"),
    ).reset_index()
    col_map = {"revenue": "revenue", "units": "units", "kg": "kg"}
    return agg.sort_values(col_map[by], ascending=False).head(n)


def abc_classification(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute the ABC/Pareto classification directly from raw data (no dependency
    on a pre-generated CSV, so this works even on a fresh clone before any notebook is run)."""
    agg = df.groupby("product").agg(
        revenue=("sales_value", "sum"),
        units=("qty", "sum"),
        kg=("mass_kg", "sum"),
        n_invoices=("doc_no", "nunique"),
    ).reset_index()
    agg = agg.sort_values("revenue", ascending=False).reset_index(drop=True)
    agg["rank"] = agg.index + 1
    agg["cum_pct_revenue"] = agg["revenue"].cumsum() / agg["revenue"].sum() * 100

    def classify(pct):
        if pct <= 80:
            return "A"
        elif pct <= 95:
            return "B"
        return "C"

    agg["abc_class"] = agg["cum_pct_revenue"].apply(classify)
    agg["kg_rank"] = agg["kg"].rank(pct=True)
    agg["volume_tier"] = np.where(agg["kg_rank"] >= 0.5, "High Volume", "Low Volume")
    return agg


def abc_summary(product_abc: pd.DataFrame) -> pd.DataFrame:
    summary = product_abc.groupby("abc_class").agg(
        n_products=("product", "count"), total_revenue=("revenue", "sum")
    ).reindex(["A", "B", "C"])
    summary["pct_of_products"] = summary["n_products"] / summary["n_products"].sum() * 100
    summary["pct_of_revenue"] = summary["total_revenue"] / summary["total_revenue"].sum() * 100
    return summary.reset_index()


def revenue_volume_quadrant(product_abc: pd.DataFrame) -> pd.DataFrame:
    q = product_abc.copy()
    q["rev_rank"] = q["revenue"].rank(pct=True)

    def quadrant(row):
        if row["rev_rank"] >= 0.5 and row["kg_rank"] >= 0.5:
            return "High Revenue / High Volume"
        if row["rev_rank"] >= 0.5 and row["kg_rank"] < 0.5:
            return "High Revenue / Low Volume (Premium)"
        if row["rev_rank"] < 0.5 and row["kg_rank"] >= 0.5:
            return "Low Revenue / High Volume (Commodity)"
        return "Low Revenue / Low Volume"

    q["quadrant"] = q.apply(quadrant, axis=1)
    return q


def rfm_segmentation(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute RFM segments directly from raw data."""
    snapshot_date = df["date"].max() + pd.Timedelta(days=1)
    rfm = df.groupby("customer").agg(
        recency=("date", lambda x: (snapshot_date - x.max()).days),
        frequency=("doc_no", "nunique"),
        monetary=("sales_value", "sum"),
    ).reset_index()

    rfm["R_score"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm["F_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)

    def assign_segment(row):
        r, f, m = row["R_score"], row["F_score"], row["M_score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        if r >= 3 and f >= 3 and m >= 3:
            return "Loyal Customers"
        if r >= 4 and f <= 2:
            return "New Customers"
        if r >= 3 and f <= 3 and m <= 3:
            return "Potential Loyalists"
        if r <= 2 and f >= 4 and m >= 4:
            return "Can't Lose Them"
        if r <= 2 and f >= 3:
            return "At Risk"
        if r <= 2 and f <= 2 and m <= 2:
            return "Hibernating"
        return "Lost Customers"

    segment_action = {
        "Champions": "Reward and retain — priority service, early access, ask for referrals.",
        "Loyal Customers": "Upsell/cross-sell — engage regularly with loyalty perks.",
        "Potential Loyalists": "Nurture — incentives to increase purchase frequency.",
        "New Customers": "Onboard well — build the relationship early.",
        "At Risk": "Win back — personalized outreach before they're lost.",
        "Can't Lose Them": "Urgent re-engagement — high value but going quiet.",
        "Hibernating": "Low-cost reactivation — win-back campaigns only.",
        "Lost Customers": "Deprioritize unless strategically important.",
    }

    rfm["segment"] = rfm.apply(assign_segment, axis=1)
    rfm["recommended_action"] = rfm["segment"].map(segment_action)
    return rfm


def moving_average_forecast(df: pd.DataFrame, horizon_days: int = 30, window: int = 7) -> pd.DataFrame:
    """Lightweight fallback forecast (no trained model / sklearn dependency) used by the
    dashboard when models/sales_forecast_model.pkl hasn't been generated yet."""
    daily = df.groupby("date")["sales_value"].sum()
    full_idx = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_idx).fillna(0)

    last_avg = daily.iloc[-window:].mean()
    future_dates = pd.date_range(daily.index.max() + pd.Timedelta(days=1), periods=horizon_days, freq="D")
    dow_avg = daily.groupby(daily.index.dayofweek).mean()
    overall_avg = daily.mean()
    dow_factor = (dow_avg / overall_avg).to_dict()

    forecast_vals = [last_avg * dow_factor.get(d.dayofweek, 1.0) for d in future_dates]
    return pd.DataFrame({"date": future_dates, "forecast_sales": forecast_vals})