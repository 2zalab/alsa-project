# Installation et Résolution de Problèmes

## Installation Rapide

### Étape 1: Installer les dépendances

```bash
pip install -r requirements.txt
```

Ou installer manuellement:

```bash
pip install numpy scipy scikit-learn nltk pandas tabulate matplotlib seaborn tqdm psutil
```

### Étape 2: Télécharger les données NLTK

```bash
python -c "import nltk; nltk.download('stopwords')"
```

### Étape 3: Vérifier l'installation

```bash
python test_simple.py
```

Si vous voyez "✓ ALL TESTS PASSED!", l'installation est réussie!

---

## Problèmes Courants et Solutions

### 1. ImportError: Missing optional dependency 'tabulate'

**Symptôme:**
```
ImportError: Missing optional dependency 'tabulate'. Use pip or conda to install tabulate.
```

**Solution:**
```bash
pip install tabulate
```

**Note:** Ce problème a été corrigé dans la dernière version. Le code inclut maintenant un fallback qui crée des tableaux Markdown manuellement si tabulate n'est pas disponible.

---

### 2. ValueError: matmul dimension mismatch

**Symptôme:**
```
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0
```

**Solution:** Ce bug a été corrigé dans le commit `95fc4fbdb`. Assurez-vous d'avoir la dernière version:
```bash
git pull origin claude/implement-adaptive-lsa-Ndewy
```

Voir [BUGFIX.md](BUGFIX.md) pour les détails complets.

---

### 3. ModuleNotFoundError: No module named 'nltk'

**Solution:**
```bash
pip install nltk
python -c "import nltk; nltk.download('stopwords')"
```

---

### 4. Problèmes de téléchargement de datasets

Si vous avez des problèmes pour télécharger les datasets (20 Newsgroups, etc.) en raison de restrictions réseau:

**Option 1: Téléchargement manuel**

1. **SMS Spam Collection:**
   - Télécharger: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
   - Extraire dans: `data/sms_spam/SMSSpamCollection`

2. **IMDb Reviews:**
   - Télécharger: https://ai.stanford.edu/~amaas/data/sentiment/
   - Extraire dans: `data/imdb/train/` et `data/imdb/test/`

3. **20 Newsgroups:**
   - Automatiquement téléchargé par scikit-learn la première fois

**Option 2: Utiliser les données de test synthétiques**

```bash
python test_simple.py  # Utilise des données synthétiques
```

---

## Vérification de l'Installation

### Test Complet

```bash
# Test unitaires
pytest tests/ -v

# Test simple avec données synthétiques
python test_simple.py

# Test sur 20 Newsgroups (nécessite connexion internet)
python experiments/run_newsgroups.py
```

### Installation Minimale pour les Tests

Si vous voulez juste tester A-LSA sans les datasets réels:

```bash
pip install numpy scipy scikit-learn pandas tabulate
python test_simple.py
```

---

## Structure des Datasets

### SMS Spam Collection
```
data/sms_spam/
└── SMSSpamCollection
```

### IMDb Reviews
```
data/imdb/
├── train/
│   ├── pos/  (12,500 fichiers .txt)
│   └── neg/  (12,500 fichiers .txt)
└── test/
    ├── pos/  (12,500 fichiers .txt)
    └── neg/  (12,500 fichiers .txt)
```

### 20 Newsgroups
Téléchargé automatiquement par scikit-learn dans `~/scikit_learn_data/`

---

## Versions Recommandées

- Python: 3.8 ou supérieur
- NumPy: 1.21.0+
- scikit-learn: 1.0.0+
- pandas: 1.3.0+
- tabulate: 0.8.9+

---

## Support

Si vous rencontrez d'autres problèmes:

1. Vérifiez que toutes les dépendances sont installées: `pip list`
2. Vérifiez la version de Python: `python --version`
3. Consultez les logs d'erreur complets
4. Vérifiez [BUGFIX.md](BUGFIX.md) pour les bugs connus et leurs solutions

---

## Installation avec Environnement Virtuel (Recommandé)

```bash
# Créer un environnement virtuel
python -m venv alsa-env

# Activer l'environnement
# Windows:
alsa-env\Scripts\activate
# Linux/Mac:
source alsa-env/bin/activate

# Installer les dépendances
pip install -r requirements.txt

# Télécharger les données NLTK
python -c "import nltk; nltk.download('stopwords')"

# Tester
python test_simple.py
```

---

## Mise à Jour

Pour obtenir les dernières corrections de bugs:

```bash
git pull origin claude/implement-adaptive-lsa-Ndewy
pip install -r requirements.txt --upgrade
```
