# SHAP Attribution Instability Under Feature Correlation: Implications for Model Risk Management

**Caraker, Arnold, Rhoads (2026)**
*Based on: "First-Mover Bias in Gradient Boosting Explanations" ([arXiv:2603.22346](https://arxiv.org/abs/2603.22346))*

---

## Executive Summary

SHAP feature importance rankings from gradient-boosted models (XGBoost, LightGBM, CatBoost) can change substantially when the model is retrained with a different random seed. This instability is mathematically inherent when input features are correlated — which is the case for the majority of real-world datasets. We provide a validated detection method and a simple, auditable fix.

---

## The Finding

When features in a model are correlated (e.g., income and debt-to-income ratio, or multiple measurements of the same physical quantity), the feature identified as "most important" by SHAP can change based solely on the model's random training seed. In our experiments:

- **On the Breast Cancer diagnostic dataset** (30 features, 21 correlated pairs): the standard SHAP workflow produces essentially random feature rankings across retrains (stability = 0.376 on a 0-1 scale).
- **Training a larger model does not fix the problem.** A model with 15,000 trees produces *less* reproducible explanations than a model with 75 trees. Sequential model construction amplifies the instability.
- **The fix is simple.** Training 25 models with different random seeds and averaging their SHAP values restores stability to 0.925+ on the same dataset.

---

## Regulatory Relevance

### SR 11-7 (Federal Reserve Model Risk Management Guidance)

Model validation under SR 11-7 requires assessment of model limitations, including sensitivity to assumptions and implementation choices. If SHAP-based feature importance is used for:

- Adverse action notices
- Model documentation
- Feature selection for production pipelines
- Ongoing model monitoring

...then the sensitivity of these explanations to the training seed is a model limitation that should be assessed and documented.

### EU AI Act, Article 13(3)(b)(ii)

High-risk AI systems must be accompanied by information about "known and foreseeable circumstances in which the AI system [...] may lead to risks to health, safety or fundamental rights." Attribution instability under feature correlation is a known and foreseeable circumstance when:

- The model uses gradient boosting (XGBoost, LightGBM, CatBoost)
- Input features include correlated predictors
- Decisions are informed by SHAP-based feature importance rankings

---

## Detection

To check whether your model's SHAP explanations are affected:

```python
pip install dash-shap
```

```python
from dash_shap import check

result = check(X_train, y_train, feature_names=feature_names)
print(result.report())
```

This trains 25 independent models, computes SHAP for each, and identifies:
- **Unstable pairs**: features whose relative ranking flips across retrains
- **Correlated groups**: features that share attribution due to correlation
- **Feature Stability Index (FSI)**: per-feature instability score

No changes to the production model are required. This is a diagnostic tool.

---

## Remediation

If instability is detected:

1. **For model documentation:** Report the DASH consensus rankings (averaged across 25 models) alongside the production model's SHAP rankings. Note which features are in unstable groups.

2. **For adverse action notices:** Use consensus rankings to determine the top adverse factors. This produces consistent explanations across retrains.

3. **For ongoing monitoring:** Add the FSI diagnostic to the model validation workflow. Re-run periodically to detect if explanation stability has changed.

DASH itself is a diagnostic tool, not a replacement for the production model. It does not change predictions, model architecture, or deployment. It changes how explanations are validated and reported.

---

## References

1. Caraker, Arnold, Rhoads (2026). "First-Mover Bias in Gradient Boosting Explanations: Mechanism, Detection, and Resolution." [arXiv:2603.22346](https://arxiv.org/abs/2603.22346)

2. Caraker, Arnold, Rhoads (2026). "The Attribution Impossibility: Faithful, Stable, and Complete Feature Rankings Cannot Coexist Under Collinearity." arXiv preprint. Lean 4 formalization: [github.com/DrakeCaraker/dash-impossibility-lean](https://github.com/DrakeCaraker/dash-impossibility-lean)

3. Software: [github.com/DrakeCaraker/dash-shap](https://github.com/DrakeCaraker/dash-shap) — `pip install dash-shap`
