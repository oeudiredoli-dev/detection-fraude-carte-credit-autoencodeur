# DÉTECTION DE FRAUDE PAR CARTE DE CRÉDIT AVEC UN AUTO-ENCODEUR

Projet d'apprentissage profond (PyTorch) : repérer les transactions frauduleuses **sans jamais voir de fraude pendant l'entraînement**, grâce à un auto-encodeur et à la détection d'anomalies.

## L'idée
Un auto-encodeur apprend à compresser puis à reconstruire une transaction. Entraîné **uniquement sur des transactions normales**, il les reconstruit bien. Une fraude, qui ne ressemble pas à ce qu'il a appris, est mal reconstruite : son **erreur de reconstruction** est élevée. On déclare une transaction suspecte quand cette erreur dépasse un seuil.

## Données
- 1 000 000 de transactions, 7 variables d'entrée et une étiquette `fraud`, sans valeur manquante.
- Taux de fraude : 8,74 %.
- Variables continues : `distance_from_home`, `distance_from_last_transaction`, `ratio_to_median_purchase_price`.
- Variables binaires : `repeat_retailer`, `used_chip`, `used_pin_number`, `online_order`.

Le fichier `credit_card_data.csv` (73 Mo) n'est pas dans le dépôt, il est trop lourd pour GitHub.
Source du jeu de données : **[AJOUTE LE LIEN ICI]**. Place le fichier dans le même dossier que le notebook.

## Méthode
1. **Découpage** : les transactions normales sont réparties en 80 % entraînement, 10 % validation, 10 % test. Le modèle s'entraîne sur 730 077 transactions normales et 0 fraude. La validation et le test reçoivent aussi des fraudes (8 740 chacun) en gardant le taux de fraude d'origine.
2. **Prétraitement** : `log1p` sur les 3 variables continues (très étalées), puis standardisation avec la moyenne et l'écart-type de l'entraînement seulement, pour éviter toute fuite de données.
3. **Modèle** : auto-encodeur dense, 7 → couche cachée → code → couche cachée → 7, activations ReLU, perte MSE. L'étiquette `fraud` sert uniquement à évaluer, jamais à entraîner.
4. **Recherche d'hyperparamètres** : recherche aléatoire (6 essais par modèle) sur la taille du code, la taille de la couche cachée et le taux d'apprentissage. Deux variantes sont comparées : un auto-encodeur classique et un auto-encodeur **débruitant** (du bruit est ajouté à l'entrée, la cible reste l'entrée propre).
5. **Entraînement final** : Adam, lots de 4096, jusqu'à 30 époques avec arrêt anticipé (patience de 5), répété avec 3 graines. Le meilleur modèle est choisi selon la **PR-AUC** sur la validation.
6. **Seuil de détection** : plusieurs percentiles de l'erreur des transactions normales (validation) sont testés. On garde le plus haut qui conserve un rappel d'au moins 75 %, donc le moins de fausses alertes pour cet objectif.

Le modèle sauvegardé a un code de 5 dimensions et une couche cachée de 64 neurones.

## Résultats sur le jeu de test
91 260 transactions normales et 8 740 fraudes.

| Mesure | Valeur |
|---|---|
| PR-AUC | 0,814 (un modèle au hasard : 0,087) |
| ROC-AUC | 0,966 |
| Rappel | 0,750 |
| Précision | 0,702 |
| F1 | 0,725 |
| Fausses alertes (normales signalées) | 3,05 % |

Matrice de confusion : 88 474 normales bien classées, 2 786 fausses alertes, 6 557 fraudes détectées, 2 183 fraudes manquées.

## Limites
- Le test contient 8,74 % de fraudes, comme le jeu de données. Dans la réalité la fraude est bien plus rare, et la précision serait alors plus faible pour le même seuil.
- Le modèle n'utilise que 7 variables, sans l'historique du client ni l'heure de la transaction.
- Le seuil est choisi pour atteindre un rappel de 75 %. Un autre objectif (moins de fausses alertes, ou plus de fraudes détectées) demande un autre seuil.

## Fichiers
- `code.ipynb` : le notebook complet (chargement, prétraitement, entraînement, évaluation, exemples de fraudes détectées, sauvegarde).
- `modele_final.pt` : poids du modèle, avec la moyenne, l'écart-type et le seuil nécessaires pour le réutiliser.

## Installation et utilisation
```
pip install torch numpy pandas matplotlib scikit-learn
```
1. Place `credit_card_data.csv` à côté de `code.ipynb`.
2. Ouvre le notebook et exécute les cellules dans l'ordre. Les graines sont fixées pour obtenir les mêmes résultats à chaque exécution.
3. Pour recharger le modèle sauvegardé, voir la dernière section du notebook : elle recrée l'auto-encodeur avec les tailles enregistrées puis charge les poids.
