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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
import pandas as pd
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
from sklearn import tree

def cm_analysis(y_true, y_pred, filename, labels, ymap=None, figsize=(10,10)):
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
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    plt.savefig(filename)

# Carregando o conjunto de dados
db = pd.read_csv('database/card_transdata.csv')

# Dividir o conjunto de dados original em treinamento e teste (80% treino, 20% teste)
X_original = db.drop(['ratio_to_median_purchase_price', 'online_order', 'fraud'], axis=1)
y_original = db['fraud']

X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(X_original, y_original, test_size=0.2, random_state=42)

# Separar as classes no conjunto de treinamento
fraud_data_train = pd.concat([X_train_original, y_train_original], axis=1)[y_train_original == 1]
non_fraud_data_train = pd.concat([X_train_original, y_train_original], axis=1)[y_train_original == 0]

# Realizar undersampling na classe majoritária do conjunto de treinamento
non_fraud_downsampled_train = resample(non_fraud_data_train, 
                                       replace=False, 
                                       n_samples=len(fraud_data_train), 
                                       random_state=42)

# Combinar as classes balanceadas no conjunto de treinamento
balanced_train_data = pd.concat([non_fraud_downsampled_train, fraud_data_train])

# Verificar a distribuição da variável alvo nos conjuntos de treinamento e teste
train_distribution = balanced_train_data['fraud'].value_counts(normalize=True)
test_distribution = y_test_original.value_counts(normalize=True)

train_distribution, test_distribution
# Preparar os conjuntos de treinamento e teste
X_train_balanced = balanced_train_data.drop('fraud', axis=1)
y_train_balanced = balanced_train_data['fraud']


# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler to the training data and transform it
X_train_scaled = scaler.fit_transform(X_train_balanced)
# Transform the test data using the same scaler
X_test_scaled = scaler.transform(X_test_original)

# Treinar o modelo de Árvore de Decisão novamente
# Definir a grade de hiperparâmetros e as distribuições
param_dist = {
    'n_estimators': [10, 20, 30, 40, 50],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [10, 20, 30, 40, 50, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
"""
# Iniciar o Random Search
modelo = RandomizedSearchCV(estimator=RandomForestClassifier(random_state=42),
                                      param_distributions=param_dist,
                                      n_iter=10, 
                                      cv=3, 
                                      verbose=1, 
                                      random_state=42, 
                                      n_jobs=-1)

# Ajustar o modelo
modelo.fit(X_train_scaled, y_train_balanced)

# Obter os melhores hiperparâmetros
best_params = modelo.best_params_
print('======================================================================================')
print(best_params)
print('======================================================================================')
"""
#modelo = RandomForestClassifier(random_state=42, n_estimators=100)
#modelo.fit(X_train_balanced, y_train_balanced)


# ...

# O restante do seu código permanece o mesmo até a parte onde o modelo é treinado

# Treinando o modelo usando Gradient Boosting
modelo_gb = GradientBoostingClassifier(random_state=3, n_estimators=10)
modelo_gb.fit(X_train_scaled, y_train_balanced)

# Fazer previsões usando o conjunto de teste com o modelo Gradient Boosting
previsoes_gb = modelo_gb.predict(X_test_scaled)

# Calculando as probabilidades da classe positiva (fraude) com o modelo Gradient Boosting
probs_gb = modelo_gb.predict_proba(X_test_scaled)[:, 1]

# Calculando a precisão e o recall para Gradient Boosting
precision_gb, recall_gb, _ = precision_recall_curve(y_test_original, probs_gb)

# Calculando o F1-score para Gradient Boosting
f1_score_gb = 2 * (precision_gb * recall_gb) / (precision_gb + recall_gb)

# Plotando os gráficos de Precisão, Recall e F1-score para Gradient Boosting
plt.figure(figsize=(10, 5))
plt.subplot(131)
plt.plot(recall_gb, precision_gb, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precisão')
plt.title('Curva de Precisão-Recall (Gradient Boosting)')

plt.subplot(132)
plt.plot(recall_gb, f1_score_gb, marker='.')
plt.xlabel('Recall')
plt.ylabel('F1-score')
plt.title('Curva de F1-score vs. Recall (Gradient Boosting)')

plt.subplot(133)
plt.plot(precision_gb, f1_score_gb, marker='.')
plt.xlabel('Precisão')
plt.ylabel('F1-score')
plt.title('Curva de F1-score vs. Precisão (Gradient Boosting)')

plt.tight_layout()
plt.savefig("images/graficos_gradient_boosting.png", format="png")

# Matriz de Confusão para Gradient Boosting
cm_gb = ConfusionMatrix(modelo_gb)
cm_gb.score(X_test_scaled, y_test_original)
print(classification_report(y_test_original, previsoes_gb))

# Salvar a matriz de confusão do Gradient Boosting em um arquivo
cm_analysis(y_test_original, previsoes_gb, 'images/confusion_matrix_gradient_boosting.png', ['0', '1'])
plt.clf()
plt.show()
