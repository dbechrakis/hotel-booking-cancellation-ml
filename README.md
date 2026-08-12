# Hotel Booking Cancellation Prediction

An end-to-end machine learning case study focused on predicting hotel booking cancellations and identifying the operational factors that drive cancellation risk.

The project uses the **Hotel Booking Demand dataset (~120K bookings)** and combines exploratory analysis, feature engineering, classification, regression, and SHAP-based explainability to translate booking data into actionable business insights.

## Business Problem

Hotel cancellations create uncertainty in inventory planning, revenue forecasting, and operational capacity.

The objective is to answer two related questions:

1. **Which bookings are most likely to be cancelled?**
2. **Which booking characteristics are associated with higher cancellation risk and longer booking lead times?**

The analysis is designed from a revenue-management perspective, with an emphasis on interpretable results and actionable signals rather than model performance alone.

## Key Results

| Task | Model | Key Metric |
|---|---|---:|
| Cancellation prediction | Logistic Regression | F1: **0.6357** |
| Cancellation prediction | Random Forest | F1: **0.6583** |
| Lead-time prediction | Ridge Regression | R²: **0.3858** |
| Lead-time prediction | Gradient Boosting | R²: **0.5833** |

The Random Forest achieved the strongest cancellation-classification performance, while Gradient Boosting substantially improved lead-time prediction compared with the linear baseline.

## Analysis Workflow

### 1. Exploratory Data Analysis

The analysis examines cancellation patterns across hotel type, market segment, lead time, seasonality, deposit type, and booking characteristics.

A key finding is the strong relationship between **lead time and cancellation behaviour**, with cancelled bookings generally being made substantially further in advance.

### 2. Feature Engineering

The modelling pipeline includes:

- Lead time
- Repeat-guest indicators
- Deposit type
- Special-request information
- Stay-duration features
- `total_guests`
- `total_nights`
- `revenue_proxy`
- Categorical encoding and numerical preprocessing

Potentially leaking post-booking variables such as `reservation_status` and `reservation_status_date` are excluded from the cancellation model.

### 3. Machine Learning

Two classification approaches are evaluated:

- Logistic Regression — interpretable baseline
- Random Forest — non-linear ensemble model

A separate regression task predicts booking lead time using:

- Ridge Regression — linear baseline
- Gradient Boosting — non-linear model

All models are evaluated on held-out test data.

### 4. Explainability

SHAP is used to understand model behaviour and identify the features contributing most strongly to predictions. This moves the analysis beyond model scores towards explanations that revenue teams can act on.

## Key Visualizations

Selected figures are stored under `figures/` and cover cancellation patterns, correlations, lead-time distributions, monthly trends, SHAP explainability, and regression diagnostics.

## Business Insights

- **Longer lead times are associated with higher cancellation risk.**
- **Market segment and booking characteristics provide meaningful information about cancellation behaviour.**
- Cancellation risk can therefore support proactive revenue-management workflows.
- High-risk bookings could potentially be targeted with differentiated cancellation policies, non-refundable incentives, or targeted retention actions.
- SHAP explanations can help revenue teams understand why an individual booking was flagged.

These findings are predictive signals, not causal conclusions. Any operational intervention should be validated with live experiments before production deployment.

## Repository Structure

```text
hotel-booking-cancellation-ml/
│
├── README.md
│
├── notebooks/
│   └── Hotel_Booking_Complete.ipynb
│
├── figures/
│   ├── plot1_cancellation_by_segment.jpg
│   ├── plot2_heatmap.jpg
│   ├── plot3_boxplot_leadtime.jpg
│   ├── plot4_monthly_trends.jpg
│   ├── regression_residuals.jpg
│   ├── shap_classification.jpg
│   └── shap_regression.jpg
│
├── docs/
│   ├── Hotel_Booking_Demand_ML_Strategy_Report.docx
│   └── Hotel_Booking_ML_Strategy.pptx
│
└── README.md
```

## Reproducibility

The complete analysis is available in `notebooks/Hotel_Booking_Complete.ipynb`.

The original **Hotel Booking Demand** dataset is not included in this repository. This keeps the repository focused on the modelling workflow and avoids duplicating a large external dataset.

## Limitations

- The dataset represents historical hotel bookings and may not generalize directly to other properties or markets.
- Model performance depends on the available booking features and historical behaviour.
- The analysis identifies predictive relationships and should not be interpreted as causal inference.
- Additional validation on more recent or property-specific data would be required before production deployment.

## Tech Stack

**Python · Pandas · NumPy · Scikit-learn · SHAP · Matplotlib · Seaborn · Jupyter**

## Author

**Dimitris Bechrakis**  
M.Sc. Data Science — The American College of Greece
