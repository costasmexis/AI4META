import shap

def _calc_shap(X_train, X_test, model):
    try:
        explainer = shap.Explainer(model, X_train)
    except TypeError as e:
        if (
            "The passed model is not callable and cannot be analyzed directly with the given masker!"
            in str(e)
        ):
            print(
                "Switching to predict_proba due to compatibility issue with the model."
            )
            explainer = shap.Explainer(lambda x: model.predict_proba(x), X_train)
        else:
            raise TypeError(e)
    try:
        if model.__class__.__name__ in ["LGBMClassifier", "CatBoostClassifier","RandomForestClassifier"]:
            shap_values = explainer(X_test, check_additivity=False)
        else: 
            shap_values = explainer(X_test)
    except ValueError:
        num_features = X_test.shape[1]
        max_evals = 2 * num_features + 1
        shap_values = explainer(X_test, max_evals=max_evals)

    if len(shap_values.shape) == 3:
        shap_values = shap_values[:, :, 1]
    else:
        pass
    return shap_values
 
