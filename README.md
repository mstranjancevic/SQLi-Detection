# SQL Injection Detection using Random Forest

Seminarski rad — detekcija SQL Injection napada pomoću mašinskog učenja.

## Opis
Model klasifikuje SQL upite kao legitimne (0) ili maliciozne (1) 
korišćenjem TF-IDF vektorizacije i Random Forest algoritma.

## Rezultati
- Accuracy: 99.64%
- F1-score: 99.62%
- Dataset: 30.597 SQL upita

## Pokretanje
1. Pokreni `main.py` za treniranje modela
2. Pokreni `test.py` za interaktivno testiranje
3. Pokreni `visual.py` za vizuelizaciju karakteristika

## Tehnologije
Python | scikit-learn | pandas | matplotlib | seaborn