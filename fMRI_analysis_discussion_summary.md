# fMRI Analysis Discussion Summary
## Comparing Activation Maps to Standard GLM 1st/2nd Level Analysis

**Date:** December 16, 2025  
**Context:** Analysis of `DFG_fit_fidge_stimXcog_maps.py` - Ridge regression on activation maps with cognitive features

---

## 1. The Question

Should we subtract the subject mean when computing activation maps?

```python
# Two options in the code:
if subtract_subject_mean:
    act = (mean_stim - mean_sub) / std_sub   # Option A: Contrast-like
else:
    act = mean_stim / std_sub                 # Option B: Beta-like
```

---

## 2. Standard fMRI GLM Analysis Overview

### 1st Level Analysis (Within-Subject)

The GLM model for each subject:
```
Y(t) = β₀ + β₁·X₁(t) + β₂·X₂(t) + ... + ε(t)
```

Where:
- **Y(t)** = BOLD signal timeseries
- **β₀** = Intercept (captures the baseline)
- **β₁, β₂, ...** = Beta coefficients for each stimulus/condition
- **X₁(t), X₂(t), ...** = Design matrix regressors (convolved with HRF)

#### What is "Baseline"?
- The baseline is **NOT explicitly modeled** as a separate regressor
- It's captured by the **intercept (β₀)**
- Represents timepoints where **no stimulus regressor is active** (X = 0)
- Called the **"implicit baseline"** (rest periods, fixation, inter-trial intervals)

#### How "Stimulus > Baseline" Contrast Works:
- When X₁(t) = 0 (no stimulus): Model predicts **Y = β₀**
- When X₁(t) = 1 (stimulus on): Model predicts **Y = β₀ + β₁**
- Therefore: **β₁ = (Y during stim) - (Y during baseline)**
- The beta coefficient itself IS the "stimulus > baseline" effect!

### 2nd Level Analysis (Between-Subject/Group)

- **Input:** Contrast maps from 1st level (NOT raw betas)
- **Purpose:** Compare contrasts across subjects with covariates (age, cognitive scores, group membership)
- **Key Point:** Uses **contrasts**, not absolute beta values, because different subjects have different baseline BOLD levels

---

## 3. Your Analysis Goal

**Goal:** Compare brain responses across subjects with different cognitive features.

This is essentially a **2nd-level-like analysis** where you:
1. Compute per-subject activation patterns (like 1st-level)
2. Model these patterns across subjects with cognitive covariates (like 2nd-level)

---

## 4. Why You SHOULD Subtract the Mean (`SUBTRACT_SUBJECT_MEAN = True`)

### Your Computation:
```python
act = (mean_stim - mean_sub) / std_sub
```

### This is Analogous to a Contrast Because:

| Your Code | GLM Equivalent |
|-----------|----------------|
| `mean_stim` | β₀ + β₁ (BOLD during stimulus) |
| `mean_sub` | ≈ β₀ (average BOLD = implicit baseline) |
| `mean_stim - mean_sub` | ≈ β₁ (the stimulus effect / contrast) |
| `/ std_sub` | Standardization for cross-subject comparison |

### Why This is Correct for Your Goal:

1. **Cross-subject comparison requires contrasts**
   - Different subjects have different baseline BOLD levels (due to scanner drift, physiology, arousal, etc.)
   - Subtracting the mean removes these subject-specific baselines
   - Makes the maps comparable across subjects

2. **Your question is about modulation**
   - You want to know: "Do subjects with higher cognitive scores show different stimulus-evoked responses?"
   - This requires comparing **relative activation** (stimulus vs baseline), not absolute BOLD

3. **Standard practice for group analysis**
   - 2nd-level fMRI analyses always use **contrasts**, not raw betas
   - Your approach mirrors this by computing contrast-like activation maps

4. **Interpretability**
   - `(mean_stim - mean_sub) / std_sub` = "stimulus effect in standard deviations relative to subject's baseline"
   - Clear, interpretable units for cross-subject comparison

---

## 5. Important Note: Your "Baseline" Definition

In your code:
```python
mean_sub = bold_sub_f.mean(axis=0)  # Mean across ALL trials for this subject
```

This is slightly different from classic GLM baseline:
- **GLM baseline:** Timepoints where design matrix = 0 (explicit rest periods)
- **Your baseline:** Mean across all stimulus conditions

This means your contrast is: **"stimulus vs. average of all conditions"** rather than **"stimulus vs. rest"**

This is still valid for your purpose because:
- It removes subject-specific baseline differences
- It focuses on relative differences between stimuli
- If you don't have explicit rest periods in your epoched data, this is a reasonable approximation

---

## 6. Summary: Your Analysis Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│  YOUR PIPELINE (Analogous to 1st + 2nd Level Combined)     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Compute Activation Maps (≈ 1st Level Contrasts)        │
│     ┌─────────────────────────────────────────────────┐    │
│     │ act = (mean_stim - mean_sub) / std_sub          │    │
│     │                                                 │    │
│     │ • Per subject, per stimulus                     │    │
│     │ • Removes subject baseline                      │    │
│     │ • Standardized (like z-scored contrast)         │    │
│     └─────────────────────────────────────────────────┘    │
│                           ↓                                 │
│  2. Ridge Regression Across Subjects (≈ 2nd Level)         │
│     ┌─────────────────────────────────────────────────┐    │
│     │ contrast_maps ~ cognitive_features × stimulus    │    │
│     │                                                 │    │
│     │ • Asks: "Do cognitive abilities predict         │    │
│     │   stimulus-evoked brain responses?"             │    │
│     └─────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Recommendation

**Keep `SUBTRACT_SUBJECT_MEAN = True`** ✓

Your current approach correctly:
- Computes contrast-like activation maps (1st level equivalent)
- Enables valid cross-subject comparison (2nd level equivalent)
- Relates brain activity to cognitive measures across subjects
- Uses standardization for interpretable, comparable units

---

## 8. Code Location

The key normalization happens in `compute_activation_maps_for_session()` (lines 280-289):

```python
if subtract_subject_mean:
    act = (mean_stim - mean_sub) / std_sub  # ← Use this for your goal
else:
    act = mean_stim / std_sub
```

Configuration flag at top of file (line 54):
```python
SUBTRACT_SUBJECT_MEAN = True  # ← Keep this True
```
