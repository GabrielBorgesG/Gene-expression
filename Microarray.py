import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import train_test_split
import csv

Nome_do_arquivo = 'Renal_GSE53757.csv'

data = pd.read_csv(Nome_do_arquivo, delimiter=',', header=0, index_col=0)
#print(data)

# a. NORMALIZAÇÃO Z = (val_obs - média)/desvio_padrao

# Mostra somente a coluna TYPE (identificando doente X controle)
#print(data['type']) # 'type' pode ser trocado pelo nome de qualquer outra coluna

# para obtermos uma lista com os identificadores das classes:
classes = data['type'].unique()
#print(classes) # ['ccRCC' 'normal']

# para acessar apenas as amostras saudáveis nesse arquivo de exemplo:
controle = data[data['type'] == 'normal']
cancer = data[data['type'] == 'ccRCC']
#print(controle) # Mostra uma tabela com as informações dos controles ~ uma matriz derivada do todo

# para obtermos a média do valor de expressão gênica de cada gene (coluna):
avg = data.mean(numeric_only=True)
#print(avg)

# para separar os dados entre as expressões em X e as classes em Y:
Y = data['type']
X = data.drop('type', axis=1)
#print(X)
#print(Y)

# scipy.stats.zscore(a, axis=0, ddof=0, nan_policy=’propagate’)

X.apply(stats.zscore)
#print(X)

# b. SUBCONJUNTOS DE TREINAMENTO E TESTE

X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.33, random_state=42)

print(X_treino)
print(X_teste)
print(Y_treino)
print(Y_teste)

Tcancer = cm[0, 0]
Fcancer = cm[1, 0]
Tnormal = cm[1, 1]
Fnormal = cm[0, 1]

# Novo ----------------------------------------------------------------------------------------------------------------
kmeans_clustering = KMeans(n_clusters=2, random_state=0).fit(X) # cria um modelo simples de k-means com 2 clusters, obtido dos dados em X
clusters = kmeans_clustering.predict(X_teste)
#print(clusters)

# d, MÉTRICAS
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

print("Matrix de confusão: ")
print(cm)

# Com função
acuracia = accuracy_score(Y_teste, prediction)
print("Acurácia: {}\n".format(acuracia))

# "Manual" para confirmar o uso correto da função
acuracia2 = (Tcancer+Tnormal)/(Tcancer + Tnormal + Fcancer + Fnormal)
#print(acuracia2)

sensibilidade = Tcancer/(Tcancer + Fnormal)
print("Sensibilidade: {}\n".format(sensibilidade))

especificidade = Tnormal/(Tnormal + Fcancer)
print("Especificidade: {}\n".format(especificidade))

print("Escore F1: ")
f1_score_macro = f1_score(Y_teste, prediction, average='macro')
f1_score_micro = f1_score(Y_teste, prediction, average='micro')
f1_score_weighted = f1_score(Y_teste, prediction, average='weighted')
print("Macro: {} | Micro: {} | Weighted: {}".format(f1_score_macro, f1_score_micro, f1_score_weighted))

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print("Escore Silhouette:  {}".format(silhouette_score(X_teste, kmeans_clustering.fit_predict(X_teste))))

# e. utilize o método k-means para analisar o dataset de estudo (preparado no item 'a')

# 2 Grupos
print("Para 2 grupos:\n{} dentro do cluster 1".format(list(clusters).count(0)))
print("{} dentro do cluster 2".format(list(clusters).count(1)))

# 3 a 5 Grupos
for i in range(3, 5):
    kmeans_clustering = KMeans(n_clusters=i, random_state=0).fit(X) # cria um modelo simples de k-means com 2 clusters, obtido dos dados em X
    clusters = kmeans_clustering.predict(X_teste)
    print("Para {} grupos:".format(i))
    for j in range(i):     
        print("{} dentro do cluster {}".format(list(clusters).count(j), j+1))
