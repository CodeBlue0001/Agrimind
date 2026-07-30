# Fix Training Issues & Test Model Errors

## Problem Analysis

After analysing [all_states_model_training.ipynb](file:///d:/researrch/agree.culture.Ai/code/model/state_crop_yeild/all_states_model_training.ipynb) and [test_model.ipynb](file:///d:/researrch/agree.culture.Ai/code/model/state_crop_yeild/test_model.ipynb), I found **3 critical issues**:

---

### Issue 1: Unnecessary LabelEncoder on a Continuous Target (Training Notebook)

In [all_states_model_training.ipynb](file:///d:/researrch/agree.culture.Ai/code/model/state_crop_yeild/all_states_model_training.ipynb) (Cell 3, the global model cell), the code does:

```python
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Encodes continuous Yield into integer labels
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, ...)  # Trains on encoded labels
```

`Yield` is a **continuous float** value (e.g., `0.796`, `5238.05`). `LabelEncoder` converts each unique float into an arbitrary integer (e.g., `0`, `1`, `2`, ...). The `RandomForestRegressor` then predicts these **meaningless integer labels** instead of actual yield values.

> [!CAUTION]
> The global model's predictions (e.g., `[86.13]`) are **encoded label indices**, not real yield values. The `crop_label_encoder.pkl` is saved but **never used to inverse-transform** predictions back in the test notebook. Even if it were, the results would be misleading since label-encoding a continuous target is semantically wrong — it destroys the ordinal relationship between yield values.

**The state-wise models (Cell 4) do NOT have this issue** — they train directly on raw `y_state = state_df['Yield']` values.

---

### Issue 2: Feature Mismatch Between Training and Testing

The **global model** was trained with these features:

| Categorical | Numerical |
|---|---|
| `Crop`, `Season`, `State` | `Production`, `Annual_Rainfall`, `Area` |

But it also received columns `Crop_Year`, `Fertilizer`, `Pesticide` in `X` (since only `Yield` was dropped). The `ColumnTransformer` silently ignores them, but they are still present in the DataFrame.

The **test notebook** provides:
```python
{'Season': 'Whole year', 'State': 'West Bengal', 'Production': 0, 'Crop': 'Coconut', 'Area': 520, 'Annual_Rainfall': 56}
```

> [!WARNING]
> - Missing columns: `Crop_Year`, `Fertilizer`, `Pesticide` — these were present during training. While the `ColumnTransformer` ignores them, their absence can cause warnings or errors depending on the sklearn version.
> - Column order differs from training (the training had `Crop` first, then `Crop_Year`, `Season`, `State`, `Area`, `Production`, `Annual_Rainfall`, `Fertilizer`, `Pesticide`).

---

### Issue 3: Test Notebook Outputs Encoded Labels, Not Real Yields

The test notebook prints:
```
[86.13]
```

This is a **label-encoded index** (from the LabelEncoder), not a real yield value. The line `model_encoder.inverse_transform(predict_encoded_data)` is commented out, and even if uncommented, would fail because the regressor returns floats like `86.13`, which `inverse_transform` expects to be integer indices.

---

## Proposed Changes

### [MODIFY] [test_model.ipynb](file:///d:/researrch/agree.culture.Ai/code/model/state_crop_yeild/test_model.ipynb)

Since the `.ipynb` format cannot be directly edited, I'll create a **clean Python script** version and then convert it. The fix will:

1. **Remove the LabelEncoder dependency** — the global model's predictions are fundamentally flawed due to label-encoding. We should instead use the **state-wise models** which predict raw yield values correctly.
2. **Create a proper test flow** that:
   - Loads the correct state-specific model based on user input
   - Provides all required features (`Crop`, `Season`, `Crop_Year`, `Area`, `Production`, `Annual_Rainfall`, `Fertilizer`, `Pesticide`)
   - Outputs the **actual predicted yield** (not an encoded label)
3. **Also support the global model** with correct feature columns (for comparison), but document that it predicts encoded labels.

> [!IMPORTANT]
> The **root cause** of the prediction error is in the training notebook's global model using `LabelEncoder` on a continuous target. The cleanest fix for the test notebook is to **use the state-wise models** which were trained correctly. Alternatively, the global model could be retrained without LabelEncoder, but that requires re-running training.

## Open Questions

> [!IMPORTANT]
> 1. **Should I retrain the global model** (remove LabelEncoder, train on raw Yield)? This would require re-running the training notebook and overwriting `crop_prediction_model.pkl`. Or should I just fix the test notebook to use the state-wise models?
> 2. The state-wise models expect `Crop_Year`, `Fertilizer`, and `Pesticide` as inputs. **Do you want placeholder/default values for these**, or should the test notebook require all fields?

## Verification Plan

### Manual Verification
- Run the fixed test notebook and confirm that the predicted yield is a **realistic value** (e.g., tonnes/hectare) rather than an encoded label index.
- Compare prediction against known values from the dataset for the same crop/state combination.
