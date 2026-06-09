import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    model = joblib.load('sqli_model_v3.pkl')
    vectorizer = joblib.load('vectorizer_v3.pkl')
    print("Model i vektorizator uspešno ucitani.")

    features = vectorizer.get_feature_names_out()
    importances = model.feature_importances_

    data = pd.DataFrame({'Feature': features, 'Importance': importances})
    data = data.sort_values(by='Importance', ascending=False).head(20)

    data['Feature'] = data['Feature'].apply(lambda x: x.replace(' ', '[sp]'))

    plt.figure(figsize=(12, 10))

    sns.barplot(
        x='Importance',
        y='Feature',
        data=data,
        hue='Feature',
        palette='magma',
        legend=False
    )

    plt.title('Top 20 najbitnijih karakteristika za detekciju SQLi', fontsize=16, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=13)
    plt.ylabel('Karakter / N-gram (Kombinacije karaktera)', fontsize=13)

    plt.grid(axis='x', linestyle='--', alpha=0.4)

    for i, v in enumerate(data['Importance']):
        plt.text(v, i, f' {v:.4f}', va='center', fontsize=10)

    plt.tight_layout()

    plt.savefig('sqli_top_20_features.png', dpi=300)
    print("Grafikon je sacuvan kao 'sqli_top_20_features.png'")

    plt.show()

except FileNotFoundError:
    print("Greska: .pkl fajlovi nisu pronadeni. Prvo pokreni trening skriptu!")
except Exception as e:
    print(f"Desila se greska: {e}")