# Bug Fix: SVD Dimension Mismatch

## Problem

The original implementation had a dimension mismatch error when trying to project documents into the latent spaces:

```
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0,
with gufunc signature (n?,k),(k,m?)->(n?,m?)
(size 10585 is different from 799)
```

## Root Cause

The issue was in how the SVD was applied to the class-specific document-term matrices:

1. **Original (Incorrect) Approach:**
   - Applied SVD on `X.T` (transposed matrix)
   - This caused `components_` to have shape `(k, n_documents)` instead of `(k, n_features)`
   - When trying to project a new document with `U^T @ x_doc`, dimensions didn't match

2. **What was happening:**
   ```python
   X_pos: (n_documents, n_features) = (799, 10585)
   X_pos.T: (n_features, n_documents) = (10585, 799)

   After SVD on X_pos.T:
   components_: (k, n_documents) = (100, 799)

   Trying to project x_doc (10585,):
   U_pos_.T @ x_doc → (799, 100)^T @ (10585,) ❌ Dimension mismatch!
   ```

## Solution

**Corrected Approach:**

1. Apply SVD directly on the document-term matrix `X` (not transposed)
2. Use `components_` which gives `V^T` of shape `(k, n_features)`
3. Project documents using `V^T @ x_doc`

```python
# Corrected code:
self.svd_pos_.fit(X_pos)  # No transpose!
self.V_pos_ = self.svd_pos_.components_  # shape: (k, n_features)

# Projection:
z_pos = (self.V_pos_ @ x_doc) / self.Sigma_pos_
```

**Why this works:**
```python
X_pos: (n_documents, n_features) = (799, 10585)

After SVD on X_pos:
components_ (V^T): (k, n_features) = (100, 10585)

Projecting x_doc (10585,):
V_pos_ @ x_doc → (100, 10585) @ (10585,) = (100,) ✓ Correct!
```

## Changes Made

### In `src/alsa.py`:

1. **fit() method (lines 204-228):**
   - Changed from `self.svd_pos_.fit(X_pos.T)` to `self.svd_pos_.fit(X_pos)`
   - Renamed `U_pos_` → `V_pos_` and `U_neg_` → `V_neg_` for clarity
   - Updated documentation

2. **_compute_semantic_distance() method (lines 264-270):**
   - Changed from `(self.U_pos_.T @ x_doc)` to `(self.V_pos_ @ x_doc)`

3. **get_latent_projections() method (lines 397-400):**
   - Updated projection calculation

4. **get_characteristic_terms() method (lines 426-429):**
   - Changed indexing from `self.U_pos_[:, 0]` to `self.V_pos_[0, :]`

### In `tests/test_alsa.py`:

- Updated test assertions to check `V_pos_.shape[0]` instead of `U_pos_.shape[1]`

## Mathematical Justification

For LSA, given document-term matrix X of shape (n_docs, n_features):

**SVD Decomposition:**
```
X ≈ U Σ V^T
where:
  U: (n_docs, k) - document space
  Σ: (k, k) - singular values
  V^T: (k, n_features) - term space
```

**Projecting a new document d with TF-IDF vector x_d:**

The projection into the k-dimensional latent space is:
```
z = V^T @ x_d
```

After normalization by singular values:
```
z_normalized = (V^T @ x_d) / Σ
```

This is exactly what the corrected code now implements.

## Testing

Verified with synthetic data:
- Training on 8 documents (4 per class)
- Successfully predicts on new documents
- Latent projections have correct dimensions
- Characteristic terms extracted correctly

## Status

✅ **Bug Fixed**
✅ **Tests Updated**
✅ **Code Committed and Pushed**

All experiments should now run without dimension errors.
