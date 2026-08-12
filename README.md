# Hotel Booking Cancellation Prediction

An end-to-end machine learning case study focused on predicting hotel booking cancellations and identifying the operational factors that drive cancellation risk.

The project uses the **Hotel Booking Demand dataset (~120K bookings)** and combines exploratory analysis, feature engineering, classification, regression, and SHAP-based explainability to translate booking data into actionable business insights.

## Business Problem

Hotel cancellations create uncertainty in inventory planning, revenue forecasting, and operational capacity.

The objective of this project is to answer two related questions:

1. **Which bookings are most likely to be cancelled?**
2. **Which booking characteristics are associated with higher cancellation risk and longer booking lead times?**

The analysis is designed from a revenue-management perspective, with an emphasis on interpretable results and actionable signals rather than model performance alone.

## Key Results

| Task | Model | Key Metric |
|---|---|---:|
| Cancellation prediction | Logistic Regression | F1: 0.636 |
| Cancellation prediction | Random Forest | F1: 0.658 |
| Lead-time prediction | Ridge Regression | R²: 0.386 |
| Lead-time prediction | Gradient Boosting | R²: 0.583 |

The Random Forest achieved the strongest cancellation-classification performance, while Gradient Boosting substantially improved lead-time prediction compared with the linear baseline.

## Analysis Workflow

### 1. Exploratory Data Analysis

The analysis examines cancellation patterns across:

- Hotel type
- Market segment
- Lead time
- Seasonality
- Deposit type
- Booking characteristics

Key findings include a strong relationship between **lead time and cancellation behaviour**, with cancelled bookings generally being made substantially further in advance.

### 2. Feature Engineering

The modelling pipeline includes:

- Lead-time features
- Repeat-guest indicators
- Deposit type
- Special-request information
- Stay-duration features
- Categorical encoding
- Numerical preprocessing

Potentially leaking post-booking variables such as `reservation_status` and `reservation_status_date` are excluded from the cancellation model.

### 3. Machine Learning

Two classification approaches are evaluated:

- Logistic Regression
- Random Forest

A separate regression task predicts booking lead time using:

- Ridge Regression
- Gradient Boosting Regression

Models are evaluated on held-out test data using appropriate classification and regression metrics.

### 4. Explainability

SHAP is used to understand model behaviour and identify the features contributing most strongly to predictions.

This allows the analysis to move beyond:

> "The model predicts cancellation."

towards:

> "Which characteristics are driving the predicted cancellation risk?"

## Key Visualizations

The selected figures are stored under `figures/` and include cancellation patterns, correlations, lead-time distributions, monthly trends, SHAP explainability, and regression diagnostics.

## Business Insights

The analysis suggests several commercially relevant signals:

- **Longer lead times are associated with higher cancellation risk.**
- **Market segment and deposit type provide meaningful information about cancellation behaviour.**
- Cancellation risk can therefore be incorporated into proactive revenue-management strategies.
- High-risk bookings could potentially be targeted with stricter cancellation policies, non-refundable incentives, or differentiated pricing strategies.
- Model explainability is important when using predictions to support operational decisions.

These findings should be treated as analytical signals rather than causal conclusions.

## Repository Structure

```text
hotel-booking-cancellation-ml/
│
├── notebooks/
│   └── Hotel_Booking_Complete.ipynb
│
├── figures/
│   ├── plot1_cancellation_by_segment.png
│   ├── plot2_heatmap.png
│   ├── plot3_boxplot_leadtime.png
│   ├── plot4_monthly_trends.png
│   ├── regression_residuals.png
│   └── shap_classification.png
│
├── docs/
│   ├── Hotel_Booking_Demand_ML_Strategy_Report.docx
│   └── Hotel_Booking_ML_Strategy.pptx
│
└── README.md
```

## Reproducibility

The complete analysis is available in `notebooks/Hotel_Booking_Complete.ipynb`.

The original **Hotel Booking Demand** dataset is not included in this repository. This avoids duplicating a large external dataset and keeps the repository focused on the analysis and modelling workflow.

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
