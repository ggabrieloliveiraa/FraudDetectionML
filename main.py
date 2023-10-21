"""
Trabalho Prático 1 - Inteligência Artificial
Ciência da Computação - Pontifícia Universidade Católica de Minas Gerais
Professor(a): Cristiane Neri

Grupo 1:
Eric Ferreira
Gabriel Oliveira
Helena Ferreira
Mateus Leal
Jonathan Douglas
"""
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from yellowbrick.classifier import ClassificationReport, ConfusionMatrix

from utils.utils import cm_analysis


def main():
    # Carregando o conjunto de dados
    db = pd.read_csv("database/card_transdata.csv")
    X = db.iloc[:, 0:7].values
    Y = db.iloc[:, 7].values

    rus = RandomUnderSampler(random_state=23)
    x_resampled, y_resampled = rus.fit_resample(X, Y)

    # Dividindo os dados em conjuntos de treinamento e teste
    x_treino, x_teste, y_treino, y_teste = train_test_split(
        x_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=23
    )

    Y_aux = np.delete(Y, rus.sample_indices_)
    X_aux = np.delete(X, rus.sample_indices_, 0)
    y_teste = np.concatenate((y_teste, Y_aux))
    x_teste = np.concatenate((x_teste, X_aux))

    # Treinando o modelo DecisionTreeClassifier
    modelo = DecisionTreeClassifier(criterion="entropy", max_depth=4)
    modelo.fit(x_treino, y_treino)

    # Realizando previsões no conjunto de teste
    previsoes = modelo.predict(x_teste)

    # Calculando as probabilidades da classe positiva (fraude)
    probs = modelo.predict_proba(x_teste)[:, 1]

    # Calculando a precisão, recall e f1-score para cada threshold
    precision, recall, thresholds = precision_recall_curve(y_teste, probs)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Plotando os gráficos de Precisão, Recall e F1-score
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")

    plt.subplot(132)
    plt.plot(thresholds, recall[:-1], "b--", label="Recall")
    plt.plot(thresholds, precision[:-1], "g-", label="Precision")
    plt.xlabel("Threshold")
    plt.legend(loc="lower left")
    plt.title("Precision-Recall vs Threshold Curve")

    plt.subplot(133)
    plt.plot(thresholds, f1_score[:-1], "r-", label="F1-score")
    plt.xlabel("Threshold")
    plt.legend(loc="lower left")
    plt.title("F1-score vs Threshold Curve")

    # Salvar os gráficos em um arquivo
    plt.savefig("images/graficos.png", format="png")

    fig, axs = plt.subplots(1, 3)
    fig.delaxes(axs[2])

    # Matriz de Confusão
    cm = ConfusionMatrix(modelo, is_fitted=True)
    cm.score(x_teste, y_teste)

    # Salvar a matriz de confusão em um arquivo
    cm_analysis(
        y_teste, previsoes, "images/confusion_matrix.png", ["Não fraude", "Fraude"]
    )

    previsores = [
        "distance_from_home",
        "distance_from_last_transaction",
        "ratio_to_median_purchase_price",
        "repeat_retailer",
        "used_chip",
        "used_pin_number",
        "online_order",
    ]

    # limpa a figura presente no plot
    plt.clf()

    # Plotando a árvore de decisão
    tree.plot_tree(
        modelo,
        feature_names=previsores,
        class_names=["Fraude", "Não fraude"],
        filled=True,
    )

    # Salvar a árvore de decisão em um arquivo
    plt.savefig("images/arvore.svg", format="svg")

    # Imprimir as estatísticas de classificação
    # print("Estatísticas de classificação:")
    # print(classification_report(y_teste, previsoes))
    visualizer = ClassificationReport(modelo, is_fitted=True)
    visualizer.score(x_teste, y_teste)
    visualizer.show("images/classification.svg", clear_figure=True)


# Função Principal
if __name__ == "__main__":
    main()
