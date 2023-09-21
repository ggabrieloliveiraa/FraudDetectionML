
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix
#import skle

db = pd.read_csv('card_transdata.csv')
X = db.iloc[:, 0:7].values
Y = db['fraud']
X_treino, X_teste, y_treino, y_teste = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state = 23)

modelo = DecisionTreeClassifier(criterion='entropy')
modelo.fit(X_treino, y_treino)

previsoes = modelo.predict(X_teste)
print(classification_report(y_teste, previsoes))

previsores = ['distance_from_home','distance_from_last_transaction','ratio_to_median_purchase_price','repeat_retailer','used_chip', 'used_pin_number','online_order']
tree.plot_tree(modelo, feature_names=previsores, class_names = ['Não fraude', 'Fraude'], filled=True)
plt.show()
