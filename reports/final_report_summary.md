# Final Report Summary

## Dataset Summary
- Selected sheet: Sheet1
- Rows: 4909
- Columns: 12
- Target variable: Release
- Time variable: Time
- Duplicate rows removed: 4

## Missing Value Report
|                                   |   0 |
|:----------------------------------|----:|
| Drug MW                           |   0 |
| Drug TPSA                         |   0 |
| Drug LogP                         |   0 |
| Polymer MW                        |   0 |
| LA/GA                             |   0 |
| Initial Drug-to-Polymer Ratio     |   0 |
| Particle Size                     |   0 |
| Drug Loading Capacity             |   0 |
| Drug Encapuslation Efficiency     |   0 |
| Solubility Enhancer Concentration |   0 |
| Time                              |   0 |
| Release                           |   0 |

## Key EDA Insights
- Time shows a positive association with Release (Pearson r=0.434).
- Drug Encapuslation Efficiency shows a negative association with Release (Pearson r=-0.087).
- LA/GA shows a negative association with Release (Pearson r=-0.054).
- Time appears to be a major driver of release kinetics, with correlation magnitude 0.434.
- Particle-size-related behaviour suggests larger values are linked to lower observed release in this dataset.
- Polymer-related descriptors appear influential enough to warrant mechanistic interpretation during formulation optimisation.

## Model Comparison
| Model             |          R2 |      RMSE |       MAE |   CV_Mean_R2 |   CV_Mean_RMSE |   CV_Mean_MAE |
|:------------------|------------:|----------:|----------:|-------------:|---------------:|--------------:|
| Random Forest     |  0.914399   | 0.0978617 | 0.0617772 |   0.890257   |       0.107582 |     0.0674903 |
| MLP Regressor     |  0.803766   | 0.14817   | 0.113393  |   0.758237   |       0.15955  |     0.121501  |
| Gradient Boosting |  0.779028   | 0.157232  | 0.122077  |   0.76016    |       0.159188 |     0.124446  |
| SVR               |  0.753493   | 0.166069  | 0.127866  |   0.715846   |       0.173164 |     0.131949  |
| Ridge             |  0.260296   | 0.287674  | 0.249683  |   0.210241   |       0.28894  |     0.248793  |
| Linear Regression |  0.260286   | 0.287676  | 0.249674  |   0.210222   |       0.288943 |     0.248783  |
| Lasso             | -0.00133212 | 0.334704  | 0.294751  |  -0.00428171 |       0.325816 |     0.286932  |

## Best Tuned Model
- Best baseline model: Random Forest
- Tuned holdout R2: 0.9146
- Tuned holdout RMSE: 0.0977
- Tuned holdout MAE: 0.0617
- Best CV score during tuning: 0.8904

## Explainability
- Time is the dominant model-derived driver of Release.
- Particle-size-related effects are sufficiently important to influence release control and sustained-delivery interpretation.
- Drug loading contributes materially to prediction, indicating formulation composition has a measurable effect on release behaviour.
- Encapsulation-efficiency-like variables appear mechanistically relevant, consistent with altered retention and diffusion behaviour.
- Polymer-related inputs rank among the important predictors, supporting their role in modulating barrier properties and release kinetics.

## Pharma Intelligence Layer
- Increasing Particle Size tends to decrease observed drug release in the current dataset.
- Higher loading correlates with lower release, subject to the formulation range represented in the source data.
- Encapsulation efficiency is associated with improved retention and slower release in this modelling workflow.
- Polymer-related descriptors materially influence release, indicating matrix composition contributes to sustained-release performance.

## Conclusion
The tuned Random Forest workflow demonstrates that machine learning can capture complex, nonlinear relationships in pharmaceutical formulation data and provide a reproducible basis for drug-release optimisation.
