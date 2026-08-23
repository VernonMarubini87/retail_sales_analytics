# Retail Sales Analytics & Forecasting

End-to-end retail/FMCG data science project covering business-question EDA, ABC/Pareto product analysis, RFM customer segmentation, and sales forecasting — built on 181,670 real invoice-line transactions.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![Status](https://img.shields.io/badge/status-complete-brightgreen)

## Business Problem

A retail/wholesale meat and FMCG business needs to answer the questions that actually drive decisions: which products and customers matter most, where is revenue concentrated, which accounts are at risk of churn, and what's coming next. This project answers all of that from **22 months of invoice history** (1,247 products, 50 customer accounts), and adds a forward revenue forecast to support planning.

## Project Structure

```
retail-sales-forecasting/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── retail_sales_cleaned.csv
├── notebooks/
│   ├── 01_eda_business_questions.ipynb    # sales/product/customer/FMCG business questions
│   ├── 02_abc_pareto_analysis.ipynb       # product ABC classification
│   ├── 03_customer_rfm_segmentation.ipynb # RFM customer segments
│   └── 04_sales_forecasting.ipynb         # forecasting model comparison + forward forecasts
├── models/                                 # generated on notebook run
│   ├── sales_forecast_model.pkl
│   ├── feature_columns.pkl
│   ├── model_comparison.csv
│   ├── forecast_30_day.csv / forecast_6_month.csv / forecast_12_month.csv
│   ├── product_abc_classification.csv
│   └── customer_rfm_segments.csv
└── reports/
    └── figures/                             # ~25 saved charts from the last run
```

Each notebook runs independently (`pandas.read_csv` on the same source file), so you can open just the one you need — or run all four in order for the full picture.

---

## 01 — EDA & Business Questions

Answers the questions a retail/FMCG stakeholder actually asks, grouped into four sets:

- **Sales performance** — total revenue (R40.7M), monthly/weekly/daily trends, strongest months (Jul/Jun/Mar) and days (Wed/Fri), +45.7% growth (first 6 months' average vs most recent 6 months), average transaction value (R17,557), total volume (1.03M kg)
- **Product analytics** — top 10/20/50 products by revenue/units/kg, declining vs growing products (90-day comparison), unusual pricing, promotion candidates
- **Customer analytics** — biggest customers, purchase frequency, inactive accounts, growing/declining accounts
- **Retail/FMCG questions** — revenue-vs-volume quadrant analysis, underpriced-product candidates, seasonal demand patterns, pre-peak stocking recommendations

## 02 — ABC / Pareto Product Analysis

Classic inventory-management segmentation — a stronger insight than "product X sold the most":

| Class | # Products | % of Catalogue | % of Revenue |
|---|---|---|---|
| A | 133 | 10.7% | 80.0% |
| B | 214 | 17.2% | 15.0% |
| C | 900 | 72.2% | 5.0% |

**Headline finding:** the top 10.7% of SKUs drive 80% of revenue. Class A products get priority stock control; Class C (72% of SKUs, 5% of revenue) are candidates for simplified ordering or portfolio rationalization. ABC class is also cross-tabbed against a high/low volume tier for a sharper inventory action plan.

## 03 — Customer Segmentation (RFM)

Every customer scored on **Recency, Frequency, Monetary** (quintile-based) and mapped to 8 standard segments — Champions, Loyal Customers, Potential Loyalists, New Customers, At Risk, Can't Lose Them, Hibernating, Lost Customers — each with a specific recommended account-management action.

**Headline finding:** 8 of 50 customers are "Champions," driving 95.4% of total revenue — an extremely concentrated customer base. 8 customers are "At Risk" or "Can't Lose Them," flagging where win-back attention should go first.

> Note: with only 50 customer accounts, quintile scoring reflects relative ranking within this customer base rather than universal RFM thresholds. The same method scales directly to a larger customer base.

## 04 — Sales Forecasting

Forecasts **daily total sales revenue**, comparing baselines, classical statistical methods, and machine learning:

| Model | MAE | RMSE | MAPE | WAPE |
|---|---|---|---|---|
| **Linear Regression** | **28,468** | **49,386** | **41.2%** | 49.0% |
| Random Forest | 27,819 | 47,675 | 44.8% | **47.9%** |
| Gradient Boosting | 28,264 | 48,185 | 46.8% | 48.6% |
| Moving Average (7-day) | 30,355 | 51,952 | 52.9% | 52.2% |
| Simple Exp. Smoothing | 33,244 | 53,870 | 59.7% | 57.2% |
| Seasonal Naive (lag-7) | 30,807 | 52,745 | 62.5% | 53.0% |
| Naive (lag-1) | 50,092 | 73,279 | 98.6% | 86.2% |

SARIMA and Prophet are included in the notebook as optional comparisons — they auto-skip with a clear message if `statsmodels`/`prophet` aren't installed, and slot into the same comparison table when they are (see `requirements.txt`).

**Result:** the best model roughly halves forecast error versus a naive baseline. A log transform of the target was essential — daily revenue is right-skewed with occasional bulk-order spikes (>R500k vs a ~R61k average) that otherwise dominate the loss.

**Forward forecasts** (iterative, retrained on full history):

| Horizon | Forecasted Total | Avg / Day |
|---|---|---|
| Next 30 days | ~R1.53M | ~R51,000 |
| Next 6 months | ~R8.05M | ~R44,000 |
| Next 12 months | ~R19.45M | ~R53,000 |

Confidence decreases with horizon — the notebook includes an explicit caveat on why (recursive forecasting compounds error; ~22 months of history isn't enough for the model to learn robust year-over-year seasonality). Treat 30-day numbers as the operating forecast, 6-month as a budgeting range, and 12-month as directional only.

---

## Getting Started

### Option A — Jupyter Notebook

```bash
git clone <your-repo-url>
cd retail-sales-forecasting
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/
```

### Option B — VS Code

1. Open the project folder in VS Code.
2. Install the **Python** and **Jupyter** extensions (if not already installed).
3. Create/select a virtual environment: `Ctrl+Shift+P` → *Python: Create Environment*.
4. Install dependencies: `pip install -r requirements.txt`
5. Open any notebook in `notebooks/` and select your environment's kernel (top-right kernel picker).
6. Run all cells: `Ctrl+Shift+P` → *Notebook: Run All Cells* (or run cell-by-cell with `Shift+Enter`).

> Notebooks use relative paths (`../data`, `../models`, `../reports/figures`) so they must be run **from inside `notebooks/`** — the default when opening a `.ipynb` directly in Jupyter or VS Code.

`statsmodels`, `prophet`, and `xgboost` are optional (see `requirements.txt`) — `04_sales_forecasting.ipynb` detects each one and skips it gracefully with a printed message if it isn't installed, so the notebook runs end-to-end either way.

## Dataset

`data/retail_sales_cleaned.csv` — invoice-line-level transaction data, one row per product per invoice: product, customer, quantities, mass (kg), sales value, and derived date/calendar fields. Date range: **2024-09-25 to 2026-07-31**.

## Extending This Project

- Forecast per top product or per top customer, not just total revenue
- Add external regressors: public holidays, promotions, month-end payroll cycles
- Add prediction intervals (quantile regression / bootstrapped residuals) for range-based planning
- Turn the RFM segments and ABC classes into a monthly-refreshed dashboard
- Automate monthly retraining as new invoice data lands

## Author

Vernon Marubini — [LinkedIn](#) · [GitHub](#)
