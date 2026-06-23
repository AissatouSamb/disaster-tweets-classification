# Disaster Tweets Classification

Système de classification automatique de tweets permettant de distinguer les vraies catastrophes des tweets ordinaires, basé sur l'évaluation comparative de 30 modèles NLP allant des approches classiques (TF-IDF) aux Transformers modernes (RoBERTa, BERTweet).

---

## Table des matières

1. [Contexte et problématique](#contexte-et-problématique)
2. [Dataset](#dataset)
3. [Méthodologie](#méthodologie)
4. [Résultats](#résultats)
5. [Structure du projet](#structure-du-projet)
6. [Présentation Canva](#Canva)
7. [Installation](#installation)
8. [Déploiement](#déploiement)
9. [Stack technique](#stack-technique)
10. [Auteur](#auteur)

---

## Contexte et problématique

En cas de catastrophe naturelle ou d'urgence, Twitter constitue une source d'information précieuse en temps réel. Cependant, parmi les milliers de tweets contenant des mots-clés comme *fire*, *earthquake* ou *flood*, beaucoup ne décrivent pas de vrais événements catastrophiques mais utilisent ces termes de manière métaphorique, humoristique ou publicitaire.

**Question centrale :** comment distinguer automatiquement, à grande échelle, les tweets décrivant de vraies catastrophes des usages non-pertinents ?

Cette problématique est cruciale pour :
- Les **services de secours** : identifier en temps réel les zones touchées
- Les **médias et autorités** : vérifier la véracité des informations
- Les **chercheurs** : analyser la propagation de l'information lors de crises

## Dataset

Le projet utilise le dataset [NLP with Disaster Tweets](https://www.kaggle.com/datasets/vstepanenko/disaster-tweets) de Kaggle.

| Caractéristique | Valeur |
|---|---|
| Tweets totaux (après dédoublonnage) | 11 224 |
| Train set | 8 979 (80%) |
| Test set | 2 245 (20%) |
| Type de classification | Binaire (disaster / non-disaster) |
| Déséquilibre des classes | 81.4% / 18.6% |

### Principaux types de catastrophes dans le dataset

| Catégorie | Nombre de tweets | Part |
|---|---|---|
| Incendies / Feux | 243 | 14.7% |
| Tempêtes / Vents | 204 | 12.3% |
| Guerres / Conflits | 175 | 10.6% |
| Évacuations / Urgences | 166 | 10.0% |
| Accidents de transport | 152 | 9.2% |

## Méthodologie

Six familles d'approches NLP ont été explorées et comparées :

| Notebook | Approche | Modèles |
|---|---|---|
| NB1 | Baselines Count + TF-IDF Word | 5 |
| NB2 | TF-IDF Char + Hybrid | 5 |
| NB3 | Naive Bayes complet + SGD + Features numériques | 5 |
| NB4 | Embeddings classiques (Word2Vec, GloVe Twitter) | 5 |
| NB5 | Méthodes ensemblistes (RF, XGB, LGBM, Voting, Stacking) | 5 |
| NB6 | Deep Learning (RoBERTa, BERTweet, DistilBERT, BiLSTM, TextCNN) | 5 |
| **Total** | | **30 modèles** |

**Protocole expérimental :**
- Validation croisée stratifiée 5-fold
- Optimisation des hyperparamètres via GridSearchCV
- Métrique principale : F1-score classe 1 (disaster)
- Métriques secondaires : Recall, Precision, Accuracy, Balanced Accuracy
- Métrique complémentaire : F2-score (pondère le recall 2×, plus adapté pour un système d'alerte)


## Résultats

### Top 10 des modèles

| Rang | Modèle | Catégorie | F1 | F2 | Recall | Precision |
|---|---|---|---|---|---|---|
| 1 | RoBERTa | Deep Learning | **0.771** | **0.796** | 0.814 | 0.732 |
| 2 | BERTweet | Deep Learning | 0.747 | 0.771 | 0.788 | 0.709 |
| 3 | DistilBERT | Deep Learning | 0.745 | 0.757 | 0.766 | 0.724 |
| 4 | Voting Classifier | Ensemblistes | 0.717 | 0.700 | 0.687 | 0.750 |
| 5 | Count + ComplementNB | TF-IDF Classique | 0.712 | 0.711 | 0.704 | 0.721 |
| 6 | CountBin + BernoulliNB | TF-IDF Classique | 0.711 | 0.685 | 0.667 | 0.761 |
| 7 | TF-IDF + LogReg | TF-IDF Classique | 0.710 | 0.745 | 0.771 | 0.657 |
| 8 | Stacking Classifier | Ensemblistes | 0.709 | 0.671 | 0.646 | 0.786 |
| 9 | TF-IDF + Num + LogReg | TF-IDF Classique | 0.708 | 0.748 | 0.778 | 0.649 |
| 10 | TF-IDF + LogReg Balanced | TF-IDF Avancé | 0.705 | 0.749 | 0.781 | 0.643 |

> Note : Le F1 est conservé comme métrique principale. Le F2-score, qui pondère le recall 2× plus que la precision, est calculé en complément car conceptuellement plus adapté à un système d'alerte de catastrophe où le coût d'un faux négatif est élevé.

### Apprentissages principaux

- Les **Transformers** (RoBERTa, BERTweet, DistilBERT) dominent avec F1 > 0.74, contre ~0.71 pour les meilleures approches classiques
- Le **déséquilibre des classes** (81/19) impose l'utilisation du F1 plutôt que de l'accuracy comme métrique principale
- Les **embeddings classiques moyennés** (Word2Vec, GloVe) sont moins performants que TF-IDF sur ce dataset
- L'écart de **+15 points de F1** entre les Transformers et les architectures pré-Transformer (BiLSTM, TextCNN) illustre la rupture méthodologique de 2018

**Modèle final retenu pour le déploiement : RoBERTa (F1 = 0.771)**

## Structure du projet
disaster-tweets-classification/

├── notebooks/

│   ├── EDA_Preprocessing.ipynb

│   ├── NB1_Baselines.ipynb

│   ├── NB2_Char_Hybrid.ipynb

│   ├── NB3_NaiveBayes_SGD.ipynb

│   ├── NB4_Embeddings.ipynb

│   ├── NB5_Ensemblistes.ipynb

│   ├── NB6_DeepLearning.ipynb

│   └── NB7_Comparaison.ipynb

├── dashboard/

│   ├── app.py

│   ├── pages/

│   ├── data/

│   └── requirements.txt

├── api/

│   ├── app.py

│   └── requirements.txt

├── data/

├── results/

├── README.md

└── .gitignore

## Présentation Canva
**le lien vers la présentation est disponible ici** : https://canva.link/qaa8bh5zps5lvmc

## Installation

### Prérequis
- Python 3.10 ou supérieur
- pip ou conda

### Cloner le repository

```bash
git clone https://github.com/aissatouSamb/disaster-tweets-classification.git
cd disaster-tweets-classification
```

### Lancer le dashboard Streamlit

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

### Exécuter les notebooks

Les notebooks sont conçus pour être exécutés sur Kaggle (NB1-NB5) et Google Colab (NB6-NB7) afin de bénéficier des ressources GPU gratuites nécessaires au fine-tuning des Transformers.

## Déploiement

### Dashboard interactif (Streamlit)

Analyse comparative des 30 modèles, visualisations interactives et exploration des résultats.

**Lien : **  https://disaster-tweets-classification.streamlit.app/ 


### Interface de prédiction (Hugging Face Spaces)

Interface Gradio permettant de tester le modèle RoBERTa fine-tuné sur de nouveaux tweets, avec analyse des mots ayant influencé la prédiction (méthode Leave-One-Out).

**Lien :** https://huggingface.co/spaces/aissatouSamb/disaster-tweets-classifier
## Stack technique

| Domaine | Outils |
|---|---|
| Langage | Python 3.10 |
| NLP classique | scikit-learn, NLTK, spaCy |
| Deep Learning | PyTorch, Transformers (Hugging Face) |
| Visualisation | Matplotlib, Seaborn, Plotly |
| Dashboard | Streamlit |
| API | Gradio |
| Hébergement | Streamlit Cloud, Hugging Face Spaces |
| Environnements d'exécution | Kaggle, Google Colab |

## Auteurs
**Boye DIBA**
**Aïssatou SAMB**

Projet réalisé dans le cadre du cours de Machine Learning  — ISEP2 2026

## Licence

Ce projet est distribué sous licence MIT.
