# Hotel Booking Cancellation Prediction

Applied Machine Learning project (M.Sc. Data Science). Predicts whether a hotel booking will be cancelled using the Hotel Booking Demand dataset (city + resort hotel, ~120K bookings).

## Approach

- **EDA**: cancellation rates by hotel type, market segment, lead time, and season.
- **Feature engineering**: lead time buckets, repeat-guest flags, deposit type, special requests.
- **Modelling**: classification model (cancel / not cancel) and a regression component, evaluated with residual analysis.
- **Explainability**: SHAP values for both the classification and regression models to identify the strongest drivers of cancellations.

## Key visuals produced

- Cancellation rate by market segment
- Correlation heatmap of booking features
- Lead-time boxplot by cancellation outcome
- Monthly booking trend
- Regression residuals plot
- SHAP summary plots (classification and regression)

## Status

Notebooks, the technical report, and the presentation deck are being added directly on GitHub (Jupyter notebooks and Office files render better via direct upload than through this pipeline).

## Author

Dimitris Bechrakis — M.Sc. Data Science, The American College of Greece
