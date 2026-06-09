import joblib

try:
    model = joblib.load('sqli_model_v3.pkl')
    vectorizer = joblib.load('vectorizer_v3.pkl')
    print("Model spreman! (Ukucaj 'exit' za kraj)")
except:
    print("Greska: Proveri da li su .pkl fajlovi u istom folderu!")
    exit()


def predvidi(upit):
    upit_tfidf = vectorizer.transform([upit])
    rezultat = model.predict(upit_tfidf)[0]
    verovatnoca = model.predict_proba(upit_tfidf).max() * 100

    status = "MALICIOZAN (SQLi)" if rezultat == 1 else "LEGITIMAN"
    print(f"\nRezultat: {status}")
    print(f"Sigurnost modela: {verovatnoca:.2f}%")


while True:
    korisnicki_input = input("\nUnesi SQL upit za proveru: ")

    if korisnicki_input.lower() == 'exit':
        break

    if korisnicki_input.strip() == "":
        continue

    predvidi(korisnicki_input)