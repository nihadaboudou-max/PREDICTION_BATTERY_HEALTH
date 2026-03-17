# Prédiction du State of Health (SoH) des batteries — LSTM

Projet de deep learning appliqué à la prédiction de l'état de santé de batteries lithium-ion à partir de mesures électriques et thermiques issues de cycles de décharge.

---

## Contexte

Les batteries sont au cœur des véhicules électriques, des systèmes de stockage stationnaire et des équipements IoT comme les lampadaires solaires. Le **SoH (State of Health)** mesure leur niveau de dégradation : 100 % correspond à une batterie neuve, et une valeur autour de 70-80 % signale généralement un besoin de remplacement ou de maintenance.

L'objectif de ce projet est d'entraîner un réseau **LSTM** capable d'estimer ce SoH à partir d'une courte séquence de mesures (tension, courant, température, SoC) relevées pendant un cycle de décharge, sans avoir accès à l'historique complet de la batterie.

---

## Structure du projet

```
battery_soh/
├── data/
│   └── battery_data.csv          # Données brutes (cycles de décharge)
├── src/
│   ├── data_loader.py            # Chargement et validation des données
│   ├── preprocessing.py          # Normalisation + fenêtres glissantes
│   ├── model.py                  # Architecture LSTM (PyTorch)
│   ├── train.py                  # Boucle d'entraînement + early stopping
│   └── evaluate.py               # Métriques et graphiques
├── tests/
│   ├── conftest.py               # Configuration pytest
│   ├── test_data_loader.py       # Tests module data_loader
│   ├── test_preprocessing.py     # Tests module preprocessing
│   ├── test_model.py             # Tests architecture LSTM
│   └── test_train_evaluate.py    # Tests intégration train + évaluation
├── models/                       # Poids sauvegardés (généré à l'exécution)
│   ├── lstm_soh.pth
│   └── scaler.pkl
├── results/
│   └── plots/                    # Graphiques générés automatiquement
├── main.py                       # Point d'entrée — pipeline complet
└── dashboard.html                # Dashboard interactif (prédiction temps réel)
```

---

## Dépendances

```
torch
numpy
pandas
scikit-learn
matplotlib
seaborn
pytest
```

Installation :

```bash
pip install torch numpy pandas scikit-learn matplotlib seaborn pytest
```

---

## Lancer le pipeline d'entraînement

```bash
python main.py
```

Ce que fait le pipeline dans l'ordre :

1. Lit et valide le CSV (`data/battery_data.csv`)
2. Normalise les features avec un `StandardScaler` ajusté uniquement sur les données d'entraînement
3. Découpe chaque cycle en fenêtres glissantes de taille W=3 bins
4. Sépare les données en train/test **par cycle** (les derniers 20% des cycles vont en test)
5. Entraîne un LSTM avec early stopping (patience=15)
6. Sauvegarde les poids dans `models/lstm_soh.pth` et le scaler dans `models/scaler.pkl`
7. Génère les graphiques d'évaluation dans `results/plots/`

Les hyperparamètres sont tous centralisés en haut de `main.py` pour faciliter les expérimentations.

---

## Lancer le dashboard

```bash
python -m http.server 8000
```

Puis ouvrir dans le navigateur :

```
http://localhost:8000/dashboard.html
```

Le dashboard propose quatre onglets :

- **Pipeline** : charger un CSV, régler les hyperparamètres, lancer et suivre l'exécution pas à pas avec console de log
- **Analyse** : distribution du SoH, corrélation tension/SoH, signaux sur un cycle, corrélations de Pearson entre variables
- **Prédiction** : cinq curseurs (tension, courant, température, SoC, numéro de cycle) → jauge SoH en temps réel avec interprétation automatique
- **Temps réel** : flux de mesures simulé toutes les 1,5 secondes avec graphique live et tableau défilant

> Les valeurs affichées par défaut au chargement du dashboard sont issues des 15 premières lignes du CSV fourni dans l'énoncé, codées en dur dans le JavaScript pour permettre une démo immédiate sans serveur backend.

---

## Lancer les tests

```bash
# Tous les tests
python -m pytest tests/ -v

# Un module spécifique
python -m pytest tests/test_data_loader.py -v
python -m pytest tests/test_preprocessing.py -v
python -m pytest tests/test_model.py -v
python -m pytest tests/test_train_evaluate.py -v

# Avec résumé des temps d'exécution
python -m pytest tests/ -v --durations=5
```

Les tests `test_train_evaluate.py` entraînent effectivement un LSTM sur un mini-dataset synthétique — compter 5 à 15 secondes supplémentaires.

---

## Logique de modélisation

### Pourquoi un LSTM ?

Un MLP classique traiterait chaque mesure de manière indépendante. Le LSTM lit la séquence bin après bin et accumule de l'information sur sa dynamique : comment la tension chute progressivement, comment le courant se comporte en fin de décharge. Cette information temporelle est révélatrice de l'état de dégradation de la batterie.

### Pourquoi des fenêtres glissantes ?

Un cycle peut contenir des dizaines ou centaines de mesures. Plutôt que de traiter tout le cycle d'un bloc (séquence trop longue, peu d'exemples), on le découpe en sous-séquences de taille W. Chaque sous-séquence devient un exemple d'entraînement indépendant avec le SoH du cycle comme cible. On multiplie ainsi le nombre d'exemples tout en apprenant des motifs locaux de la décharge.

Les fenêtres sont créées **à l'intérieur de chaque cycle séparément** pour éviter qu'une fenêtre enjambe deux cycles différents, ce qui serait physiquement incohérent.

### Pourquoi séparer train/test par cycle et non aléatoirement ?

Une séparation aléatoire des lignes introduirait une fuite temporelle : le modèle verrait des mesures de cycles futurs pendant l'entraînement, ce qui gonfle artificiellement les performances. En réservant les derniers 20% des cycles pour le test, on simule la situation réelle : le modèle est entraîné sur l'historique passé et évalué sur des cycles qu'il n'a jamais vus.

### Pourquoi normaliser uniquement sur le train ?

Le `StandardScaler` est ajusté (fit) sur les données d'entraînement, puis appliqué (transform) sans ré-ajustement sur le test. Si on l'ajustait sur l'ensemble complet, les statistiques du test (moyenne, écart-type) contamineraient l'entraînement — c'est ce qu'on appelle le *data leakage*.

---

## Métriques d'évaluation

| Métrique | Interprétation |
|----------|----------------|
| **MAE** | Erreur absolue moyenne en % de SoH. "En moyenne, le modèle se trompe de X%." |
| **RMSE** | Toujours ≥ MAE. Si RMSE >> MAE, certaines erreurs ponctuelles sont grandes. |
| **R²** | 1.0 = parfait. 0.0 = aussi bien qu'une constante à la moyenne. < 0 = inutile. |

---

## Résultats obtenus (dataset complet — 29 180 lignes, 24 batteries, 197 cycles)

```
MAE  : ~0.48 %
RMSE : ~0.61 %
R²   : ~0.97
```

Ces résultats indiquent que le modèle capture bien la dynamique de dégradation et prédit le SoH avec une erreur absolue inférieure à 0.5% en moyenne.

> Note : le dataset contient 140 valeurs de SoH supérieures à 100%, correspondant à des batteries mesurées en début de vie sur certains bancs de test. Ces valeurs sont signalées par `verifier_qualite()` mais non supprimées — les conserver ou les filtrer est un choix à faire en fonction du contexte industriel.

---

## Questions de réflexion (TP)

**Pourquoi le SoC est-il une variable clé pour estimer le SoH ?**
Le SoC décrit la position courante dans le cycle de décharge. À SoC identique, une batterie dégradée montrera une tension plus basse qu'une batterie neuve. Le SoC est donc une "référence" qui permet au modèle de comparer les mesures à ce qu'elles devraient être.

**Quel intérêt de découper un cycle en plusieurs fenêtres ?**
On passe de N exemples (un par cycle) à N×(L-W+1) exemples où L est le nombre de bins par cycle. Cela enrichit l'apprentissage et permet au modèle de reconnaître des motifs caractéristiques quelle que soit la phase du cycle.

**Que se passerait-il si la fenêtre était trop courte ou trop longue ?**
Trop courte (W=1 ou 2) : le modèle n'a pas assez de contexte pour capturer la dynamique temporelle — il se comporte comme un MLP. Trop longue : les fenêtres enjambent des transitions importantes et les premières mesures de la séquence deviennent trop éloignées temporellement pour être mémorisées efficacement par le LSTM.

**Quels risques si les cycles sont mal répartis en train/test ?**
Si les cycles sont répartis aléatoirement, le modèle voit pendant l'entraînement des mesures de cycles proches (dans le temps) de ceux du test. Il apprend à interpoler plutôt qu'à extrapoler sur des batteries vieillies, ce qui surestime les performances réelles.

**Dans quels cas industriels ce modèle est pertinent ?**
Supervision en temps réel de parcs de batteries (véhicules électriques, stockage solaire), maintenance prédictive pour anticiper les remplacements, optimisation de la gestion d'énergie en fonction de l'état réel des cellules.