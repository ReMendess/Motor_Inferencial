"""Treina o modelo de classificação de setores (TF-IDF + Regressão Logística).

Uso:
    python treinar_modelo.py

Gera ``modelos/modelo.joblib`` e ``modelos/vectorizer.joblib`` a partir dos
exemplos em ``dados/exemplos_treino.json``.
"""
from __future__ import annotations

from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict

from motor_inferencia import (
    EXEMPLOS_PATH,
    carregar_exemplos,
    normalizar_texto,
    treinar_modelo,
)


def main():
    dados, labels = carregar_exemplos(EXEMPLOS_PATH)
    print(f"Exemplos de treino: {len(dados)}")
    print(f"Distribuição por setor: {dict(Counter(labels))}")

    clf, vec, _, _ = treinar_modelo(EXEMPLOS_PATH, salvar=True)

    X = vec.transform([normalizar_texto(t) for t in dados])
    clf_cv = LogisticRegression(max_iter=1000, class_weight="balanced")
    pred = cross_val_predict(clf_cv, X, labels, cv=3)

    print("\nValidação cruzada (3 folds):")
    print(classification_report(labels, pred, zero_division=0))
    print("Modelo salvo em 'modelos/modelo.joblib' e 'modelos/vectorizer.joblib'.")


if __name__ == "__main__":
    main()