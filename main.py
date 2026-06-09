import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

warnings.filterwarnings("ignore")

# ============================================================
# 1. UCITAVANJE I CISCENJE PODATAKA
# ============================================================
base_path = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_path, "SQLiV3.csv")

try:
    df = pd.read_csv(file_path, encoding='utf-8-sig')
except FileNotFoundError:
    print(f"Greska: Fajl 'SQLiV3.csv' nije pronađen u folderu:\n{base_path}")
    exit()

df = df.iloc[:, :2]
df.columns = ['Query', 'Label']
df['Label'] = pd.to_numeric(df['Label'], errors='coerce')
df = df.dropna(subset=['Query', 'Label']).drop_duplicates()
df['Label'] = df['Label'].astype(int)

# ============================================================
# 2. ANALIZA DATASETA (EDA)
# ============================================================
print("=" * 50)
print("        ANALIZA DATASETA")
print("=" * 50)
print(f"Ukupno uzoraka:     {len(df)}")
print(f"Legitimni upiti:    {(df['Label'] == 0).sum()}")
print(f"Maliciozni upiti:   {(df['Label'] == 1).sum()}")
print(f"\nProcentualno:")
print(df['Label'].value_counts(normalize=True).mul(100).round(2).astype(str) + '%')
print("=" * 50)

# ============================================================
# 3. PODELA NA TRENING I TEST SET
# ============================================================
X = df['Query'].astype(str)
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTrening set: {len(X_train)} uzoraka")
print(f"Test set:    {len(X_test)} uzoraka")

# ============================================================
# 4. TF-IDF VEKTORIZACIJA
# ============================================================

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

# ============================================================
# 5. TRENIRANJE MODELA
# ============================================================
model = RandomForestClassifier(
    n_estimators=100,
    n_jobs=-1,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train_tfidf, y_train)

# ============================================================
# 6. CROSS-VALIDACIJA (5-fold)
# ============================================================
print("\n" + "=" * 50)
print("        CROSS-VALIDACIJA (5-fold)")
print("=" * 50)
cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring='f1')
print(f"F1 po foldovima: {[round(s, 4) for s in cv_scores]}")
print(f"Average F1:     {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ============================================================
# 7. CUVANJE MODELA
# ============================================================
joblib.dump(model,      os.path.join(base_path, 'sqli_model_v3.pkl'))
joblib.dump(vectorizer, os.path.join(base_path, 'vectorizer_v3.pkl'))
print("\nModel sačuvan: sqli_model_v3.pkl")
print("Vektorizator sačuvan: vectorizer_v3.pkl")

# ============================================================
# 8. EVALUACIJA NA TEST SETU
# ============================================================
y_pred = model.predict(X_test_tfidf)

print("\n" + "=" * 50)
print("        EVALUACIJA NA TEST SETU")
print("=" * 50)
print(f"Tačnost (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print("\nDetaljan izveštaj:")
print(classification_report(y_test, y_pred, target_names=['Legitiman', 'SQLi']))

# ============================================================
# 9. MATRICA KONFUZIJE
# ============================================================
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=['Legitiman', 'SQLi'],
    yticklabels=['Legitiman', 'SQLi']
)
plt.title('Matrica konfuzije – Random Forest + TF-IDF')
plt.ylabel('Stvarna klasa')
plt.xlabel('Predvidjena klasa')
plt.tight_layout()
plt.savefig(os.path.join(base_path, 'matrica_konfuzije.png'), dpi=150)
plt.show()
print("\nMatrica konfuzije sacuvana kao: matrica_konfuzije.png")