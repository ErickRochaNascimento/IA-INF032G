import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

#  CARREGAR DADOS

df = pd.read_csv("dataset_files/data.csv").sample(frac=0.1, random_state=42)

df['target'] = df['final_result'].apply(
    lambda x: 1 if x == "Withdrawn" else 0
)


X = df.drop(columns=['final_result', 'target'])
X = pd.get_dummies(X, drop_first=True)

y = df['target']

#  TREINO / TESTE

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#  NORMALIZAÇÃO (para KNN e SVM)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MODELOS

knn = KNeighborsClassifier(n_neighbors=5)
rf = RandomForestClassifier(n_estimators=100)
svm = SVC(kernel='rbf')

# Ensemble
ensemble = VotingClassifier(
    estimators=[
        ('knn', knn),
        ('rf', rf),
        ('svm', svm)
    ],
    voting='hard'
)

# TREINAR E AVALIAR

modelos = {
    "KNN": knn,
    "Random Forest": rf,
    "SVM": svm,
    "Ensemble": ensemble
}

for nome, modelo in modelos.items():
    print(f"\n===== {nome} =====")

    if nome in ["KNN", "SVM", "Ensemble"]:
        modelo.fit(X_train_scaled, y_train)
        y_pred = modelo.predict(X_test_scaled)
    else:
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

    print(classification_report(y_test, y_pred))
