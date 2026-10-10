Put here the content of `modeling_outputs.zip` downloaded from Kaggle:

    artifacts/
      meta.json
      results.csv
      test_predictions.csv
      train.csv
      test.csv
      requirements_versions.txt
      models/   (decision_tree.joblib, knn.joblib, logistic_regression.joblib, random_forest.joblib, xgboost.joblib)

If startup reports `input stream corrupted` for `xgboost.joblib`, delete only that
file and copy a fresh `xgboost.joblib` from the original Kaggle
`modeling_outputs.zip`. Keep the archive contents directly under `artifacts/`
and use the versions listed in `requirements_versions.txt`.
