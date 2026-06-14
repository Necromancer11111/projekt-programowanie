#  Projekt: Detekcja Nadużyć Finansowych w Sektorze Ochrony Zdrowia (Healthcare Fraud Detection)

**Przedmiot:** Programowanie 2
**Typ zadania:** Klasyfikacja binarna i uczenie maszynowe (Machine Learning)

---

## 1. Wprowadzenie i Kontekst Badawczy
Nadużycia i wyłudzenia finansowe (ang. *healthcare fraud*) stanowią jedno z najpoważniejszych wyzwań ekonomicznych w systemach opieki zdrowotnej. Projekt ten ma na celu zbudowanie i optymalizację klasyfikatora opartego na uczeniu maszynowym, który na podstawie profilu pacjenta, specyfiki placówki medycznej oraz parametrów finansowych roszczenia jest w stanie precyzyjnie identyfikować transakcje o wysokim ryzyku nadużycia.

## 2. Opis Zbioru Danych i Przetwarzanie (Data Engineering)
Analiza opiera się na zbiorze `healthcare_fraud_detection.csv`, liczącym **18 764 obserwacje** oraz **20 atrybutów** o charakterze numerycznym i kategorycznym.

* **Zmienna docelowa (Target):** `Is_Fraud` (0 - roszczenie uczciwe, 1 - oszustwo).
* **Imbalans klas:** Oszustwa stanowią jedynie około **8.6%** zbioru. 
* **Czyszczenie danych (Data Cleaning):** Braki w zmiennych kategorycznych (`Insurance_Type`, `Provider_Specialty`) uzupełniono nową etykietą `'Unknown'`. Braki w zmiennej numerycznej dotyczącej wcześniejszych wizyt (`Prior_Visits_12m`) poddano imputacji za pomocą mediany, ze względu na jej odporność na wartości skrajne (outliery).
* **Inżynieria cech:** Zmienne tekstowe poddano transformacji binarnej za pomocą *One-Hot Encoding* (`pd.get_dummies` z `drop_first=True`), a zbiór podzielono na treningowy (80%) i testowy (20%) z użyciem stratyfikacji (`stratify=y`), aby zachować rygorystyczne proporcje oszustw w obu podzbiorach.

## 3. Metodologia i Modele ML
Zgodnie z literaturą dotyczącą danych tabelarycznych (m m.in. T. Hastie, *The Elements of Statistical Learning*), do zadania wybrano algorytmy oparte na drzewach decyzyjnych. 
* Do ewaluacji (ze względu na imbalans) odrzucono miarę ogólnej celności (Accuracy), opierając ocenę na **Macierzy Błędu** oraz wskaźniku **F1-score dla klasy mniejszościowej**. 
* Oba modele trenowano z parametrem `class_weight='balanced'`. 
* Optymalizację hiperparametrów przeprowadzono metodą przeszukiwania siatki z walidacją krzyżową (`GridSearchCV`).

## 4. Ewaluacja i Porównanie Algorytmów

### Model Bazowy: Drzewo Decyzyjne (Decision Tree)
* **F1-score (Klasa 1):** 0.85
* Pojedyncze drzewo decyzyjne osiągnęło dobrą czułość (0.90), jednak ujawniło typową dla siebie tendencję do przeuczenia i wysokiej wariancji. Model wygenerował **93 fałszywe alarmy (False Positives)**, niesłusznie oznaczając uczciwe transakcje jako oszustwa, co skutkowało precyzją na poziomie 0.80.

### Model Zespołowy: Las Losowy (Random Forest) - Wybór Optymalny
* **F1-score (Klasa 1):** 0.89
* Algorytm (100 estymatorów, max_depth: 20) zdołał zachować niemal identyczną czułość, przy drastycznej redukcji błędów. Zastosowanie procesu *Baggingu* (agregacji bootstrapowej) pozwoliło zmniejszyć liczbę fałszywych alarmów do zaledwie **45 pomyłek** (redukcja o ponad 50% względem DT). 
* Zwiększenie precyzji do 0.89 doprowadziło do zbalansowania obu miar i osiągnięcia wyższego ostatecznego wyniku F1-score. **Random Forest został ostatecznie wybrany jako model nadrzędny w tym projekcie.**

## 5. Wyjaśnialność Modelu (Explainable AI / Feature Importance)
W celu interpretacji logiki decyzyjnej zwycięskiego algorytmu, obliczono ważność cech za pomocą miary **Mean Decrease Impurity (MDI)**. Analiza ta otwiera "czarną skrzynkę" modelu zespołowego.

**Kluczowe predyktory oszustw:**
1. **Wymiar finansowy:** Zmienne `Claim_Amount` (kwota roszczenia) oraz `Approved_Amount` (kwota zatwierdzona) stanowią absolutny fundament decyzyjny (ponad 51% zdolności predykcyjnej modelu).
2. **Wymiar behawioralno-czasowy:** Liczba dni między usługą a zgłoszeniem (`Days_Between_Service_and_Claim`) okazała się trzecim najważniejszym wskaźnikiem, co potwierdza hipotezę, że anomalie czasowe (nagłe przyspieszenia lub podejrzane opóźnienia w raportowaniu) są wskaźnikiem wyłudzeń.
3. **Wymiar demograficzny:** Wiek pacjenta (`Patient_Age`) zauważalnie wpływa na ryzyko, co sugeruje częstsze celowanie oszustów w określone grupy wiekowe.

## 6. Struktura Repozytorium
* `healthcare_fraud_detection.csv` - Surowy zbiór danych wykorzystany do analizy.
* `Projekt_P2_FraudDetection.ipynb` - Główny notatnik (Jupyter Notebook / Google Colab) zawierający pełen kod w języku Python: od czyszczenia danych (EDA), przez trening modeli, optymalizację Grid Search, aż po analizę ważności cech.
