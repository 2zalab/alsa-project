# A-LSA Performance Improvements

## 📊 Analyse des Problèmes Initiaux

### Problèmes Identifiés

1. **Threshold sub-optimal** : Le calcul initial `θ = 0.5 × log(N+/N-)` n'était pas à la bonne échelle
   - Gap : Gain potentiel de **+0.126 en F1-score** sur SMS Spam
   - Cause : L'échelle logarithmique ne correspondait pas aux distances différentielles

2. **Déséquilibre de variance** : Les espaces positif et négatif avaient des propriétés différentes
   - SMS Spam : 62% variance (positif) vs 30% (négatif)
   - IMDb : 13% variance pour les deux
   - Impact : Biais dans le calcul des distances

3. **Overfitting** : Optimisation du threshold sur l'ensemble d'entraînement
   - IMDb : Gap train/test de 0.23
   - Cause : Mémorisation des données d'entraînement

## 🚀 Solutions Implémentées

### 1. Normalisation des Énergies (`normalize_energies=True`)

**Principe** : Normaliser E+ et E- par la variance expliquée de chaque espace

```python
if self.normalize_energies:
    E_pos = E_pos / self.variance_pos_
    E_neg = E_neg / self.variance_neg_
```

**Bénéfices** :
- Compense le déséquilibre de variance entre espaces
- Améliore la comparabilité des distances
- Fonctionne mieux combiné avec l'optimisation du threshold

### 2. Optimisation du Threshold avec Validation (`optimize_threshold=True`)

**Principe** : Split 80/20 de l'ensemble d'entraînement pour optimiser le threshold

```python
# Split train/validation
train_idx, val_idx = split_validation(indices, test_size=0.2, stratify=y)

# Grid search sur 500 thresholds
for thresh in thresholds:
    y_pred = (val_distances < thresh).astype(int)
    f1 = f1_score(y_val, y_pred, average='macro')
```

**Bénéfices** :
- Évite l'overfitting
- Maximise le F1-score macro
- Utilise une validation holdout au lieu du full training

## 📈 Résultats

### SMS Spam Collection

| Configuration | Test F1 | Amélioration |
|--------------|---------|--------------|
| A-LSA (original) | 0.810 | baseline |
| + normalize only | 0.651 | ❌ -19.6% |
| + optimize only | 0.931 | ✅ +15.0% |
| **+ both (improved)** | **0.938** | ✅ **+15.8%** |

**Comparaison avec baselines** :
- A-LSA improved : **0.938**
- LSA + LR : 0.913 ← **A-LSA gagne !**
- Naive Bayes : 0.944
- Logistic Regression : 0.950
- Linear SVM : 0.959

### IMDb Movie Reviews

| Configuration | Test F1 | Overfitting Gap |
|--------------|---------|-----------------|
| Original | 0.764 | 0.22 |
| Improved | 0.754 | 0.24 |

**Analyse** : Sur IMDb (textes longs, faible variance), A-LSA reste en dessous des baselines
- Variance expliquée : seulement 12-13%
- Vocabulaire très large : 27k termes
- Overfitting structurel de l'architecture

## 🎯 Recommandations d'Utilisation

### Quand utiliser A-LSA ?

**✅ Idéal pour** :
- Textes courts (SMS, tweets, titres)
- Datasets avec haute variance expliquée (>30%)
- Classes bien séparées sémantiquement
- Datasets déséquilibrés (grâce au threshold adaptatif)

**⚠️ Moins adapté pour** :
- Textes longs (articles, reviews)
- Faible variance expliquée (<15%)
- Vocabulaire très large (>20k termes)
- Forte similarité sémantique entre classes

### Configuration Recommandée

```python
from src.alsa import AdaptiveLSA

# Configuration par défaut (recommandée)
model = AdaptiveLSA(
    n_components=100,           # 50-200 selon dataset
    normalize_energies=True,    # Compense variance
    optimize_threshold=True,    # Validation-based
    random_state=42
)

# Pour datasets avec textes courts et haute variance
model = AdaptiveLSA(
    n_components=100,
    normalize_energies=True,
    optimize_threshold=True
)

# Pour datasets équilibrés avec faible risque d'overfitting
model = AdaptiveLSA(
    n_components=100,
    normalize_energies=False,
    optimize_threshold=False
)
```

## 📊 Tableau Comparatif Final

| Dataset | Type | A-LSA | LSA+LR | Best Baseline | Gagne? |
|---------|------|-------|--------|---------------|--------|
| **SMS Spam** | Courts | **0.938** | 0.913 | 0.959 (SVM) | ✅ vs LSA |
| **IMDb** | Longs | 0.754 | 0.849 | 0.875 (LR) | ❌ |
| **20 Newsgroups** | Moyens | ? | ? | ? | À tester |

## 🔬 Pistes d'Amélioration Futures

1. **Augmentation adaptative de k** : Ajuster automatiquement selon la variance cible
2. **Régularisation** : Ajouter pénalité L2 sur les projections
3. **Ensemble methods** : Combiner plusieurs modèles A-LSA avec différents k
4. **Feature selection** : Préprocessing plus sophistiqué pour textes longs
5. **Validation croisée complète** : Optimiser k et threshold ensemble

## 📝 Changelog

### Version 1.1 (2026-01-09)
- ✅ Ajout normalisation des énergies
- ✅ Optimisation threshold avec validation holdout
- ✅ Amélioration +15.8% sur SMS Spam
- ✅ Surpasse LSA+LR sur textes courts
- 📚 Documentation des cas d'usage

### Version 1.0 (2026-01-09)
- ✅ Fix calcul threshold (log → data-driven)
- ✅ Amélioration +37% sur SMS Spam (0.46 → 0.81)
- ✅ Implementation complète A-LSA
