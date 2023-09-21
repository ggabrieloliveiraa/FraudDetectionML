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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import tree
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from yellowbrick.classifier import ConfusionMatrix


def cm_analysis(y_true, y_pred, filename, labels, ymap=None, figsize=(10, 10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args:
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    cm = confusion_matrix(y_true, y_pred)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = "%.1f%%\n%d/%d" % (p, c, s)
            elif c == 0:
                annot[i, j] = ""
            else:
                annot[i, j] = "%.1f%%\n%d" % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = "Actual"
    cm.columns.name = "Predicted"
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt="", ax=ax)
    plt.savefig(filename)


# Carregando o conjunto de dados
db = pd.read_csv("card_transdata.csv")
X = db.iloc[:, 0:7].values
Y = db["fraud"]

# Dividindo os dados em conjuntos de treinamento e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, Y, test_size=0.5, stratify=Y, random_state=23
)

# Treinando o modelo DecisionTreeClassifier
modelo = DecisionTreeClassifier(criterion="entropy", max_depth=4)
modelo.fit(X_treino, y_treino)

# Realizando previsões no conjunto de teste
previsoes = modelo.predict(X_teste)

# Calculando as probabilidades da classe positiva (fraude)
probs = modelo.predict_proba(X_teste)[:, 1]

# Calculando a precisão e o recall
precision, recall, _ = precision_recall_curve(y_teste, probs)

# Calculando o F1-score
f1_score = 2 * (precision * recall) / (precision + recall)

# Plotando os gráficos de Precisão, Recall e F1-score
plt.figure(figsize=(10, 5))
plt.subplot(131)
plt.plot(recall, precision, marker=".")
plt.xlabel("Recall")
plt.ylabel("Precisão")
plt.title("Curva de Precisão-Recall")

plt.subplot(132)
plt.plot(recall, f1_score, marker=".")
plt.xlabel("Recall")
plt.ylabel("F1-score")
plt.title("Curva de F1-score vs. Recall")

plt.subplot(133)
plt.plot(precision, f1_score, marker=".")
plt.xlabel("Precisão")
plt.ylabel("F1-score")
plt.title("Curva de F1-score vs. Precisão")

plt.tight_layout()
plt.savefig("images/graficos.png", format="png")
# Matriz de Confusão
cm = ConfusionMatrix(modelo)
cm.score(X_teste, y_teste)
print(classification_report(y_teste, previsoes))
# Salvar a matriz de confusão em um arquivo
cm_analysis(y_teste, previsoes, "images/confusion_matrix2.png", ["0", "1"])
previsores = [
    "distance_from_home",
    "distance_from_last_transaction",
    "ratio_to_median_purchase_price",
    "repeat_retailer",
    "used_chip",
    "used_pin_number",
    "online_order",
]
plt.clf()
tree.plot_tree(
    modelo, feature_names=previsores, class_names=["Não fraude", "Fraude"], filled=True
)
# plt.show()
plt.savefig("images/arvore.svg", format="svg")
