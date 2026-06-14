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

