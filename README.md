# Customer Segmentation and Prediction

Week 11 - Advanced Machine Learning Models

Uses clustering to segment 500 customers, builds a Random Forest churn model for each segment with enough data, tunes the models with Grid Search, and turns the results into business recommendations.

## Results at a glance
| Segment | Customers | Churn rate | Share of lost revenue |
|---|---|---|---|
| Premium Flexible | 100 | 23.0% | 51.1% |
| Premium Committed | 162 | 8.6% | 32.1% |
| Budget Flexible | 156 | 8.3% | 12.0% |
| Budget Committed | 82 | 3.7% | 4.8% |

- **Clustering:** K-Means, Hierarchical (Ward) and DBSCAN compared; K-Means (k = 4) chosen (silhouette 0.453).
- **Models:** separate Random Forests for the 3 segments with at least 10 churners, tuned with nested cross-validation and Grid Search; accuracy, precision, recall, F1 and ROC-AUC reported for each.
- **Key finding:** all churn happens in the first year. Models using Tenure score very high but Tenure may reveal the outcome; models using only sign-up information do no better than guessing. Segments are best used to decide who to contact first, not as a score for new customers.

## Files
- `customer_segmentation.ipynb` - the analysis notebook
- `segmentation_data.csv` - the 500 customers with their cluster labels and segment names
- `segment_profiles.md` - how the segments were made, their profiles, models, tuning and testing
- `model_evaluation_results.csv` - evaluation metrics for every model (the accuracy and counts of the tuned rows are calculated from the precision and recall printed in the notebook)
- `business_recommendations.pdf` - recommendations and impact estimates for each segment
- `requirements.txt` - Python packages
- charts: `elbow_silhouette.png`, `kmeans_clusters.png`, `dendrogram.png`, `algorithm_comparison.png`, `segment_profiles.png`, `segment_models_auc.png`, `tuning_results.png`, `business_impact.png`

## Setup
```bash
pip install -r requirements.txt
```
1. Open `customer_segmentation.ipynb` in Google Colab or Jupyter.
2. Upload `customer_churn.csv` when the first cell asks for it (the notebook looks for a file name containing "churn" or "customer").
3. Run all cells from top to bottom. The tuning step takes a few minutes.
