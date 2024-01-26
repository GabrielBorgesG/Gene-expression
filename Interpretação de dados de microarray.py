# BIBLIOTECAS PRINCIPAIS
import pandas as pd
import numpy as np
import csv
# NOTE QUE AS DEMAIS BIBLIOTECAS SERÃO INSERIDAS COM O DECORRER DO PROGRAMA A FIM DE FICAR CLARO
# EM QUE PONTO CADA UMA ESTÁ SENDO EXIGIDA (SEMPRE ABAIXO DO TÍTULO TÓPICO)

Nome_do_arquivo = 'Renal_GSE53757.csv' # Entrada do nome do arquivo
#Nome_do_arquivo = input('Nome do arquivo de entrada: ')

data = pd.read_csv(Nome_do_arquivo, delimiter=',', header=0, index_col=0)
print(data)

# a. NORMALIZAÇÃO
import scipy.stats as stats

# para obtermos uma lista com os identificadores das classes:
classes = data['type'].unique() # print(classes) = ['ccRCC' 'normal'] Neste caso

# para acessar apenas as amostras saudáveis nesse arquivo de exemplo:
controle = data[data['type'] == 'normal']
cancer = data[data['type'] == 'ccRCC']
# print(controle) # Mostra uma tabela com as informações dos controles ~ uma matriz derivada do todo

# para obtermos a média do valor de expressão gênica de cada gene (coluna):
avg = data.mean(numeric_only=True)
#print(avg)

# Para separar os dados entre as expressões em X e as classes em Y:
Y = data['type']
X = data.drop('type', axis=1)
print("{}\n{}".format(X, Y))

# Normalização dos valores obtidos
X.apply(stats.zscore)

# b. SUBCONJUNTOS DE TREINAMENTO E TESTE
from sklearn.model_selection import train_test_split

X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size=0.33, random_state=42)

print("X treino:\n{}\n".format(X_treino))
print("X teste:\n{}\n".format(X_teste))
print("Y treino:\n{}\n".format(Y_treino))
print("Y teste:\n{}\n".format(Y_teste))

# c. CLASSIFICADOR SVM
from sklearn import svm

clf = svm.SVC() # cria um modelo simples de SVM
clf.fit(X_treino, Y_treino) # treina o SVM nos pares de entrada X e Y

# clf.predict(W) # Para predizermos a classe de uma amostra desconhecida W
predicao = clf.predict(X_teste)
print("Predição:\n")
print(predicao) # mostra as classes preditas para X pelo modelo em clf
print("\n")
# para usar o kernel trick devemos passar o kernel como parâmetro na criação do SVM. Um kernel popular é o rbf (Radial Basis Function)
# rbf_svc = svm.SVC(kernel='rbf')
# rbf_svc.fit(X, Y)

# d, MÉTRICAS
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

cm = confusion_matrix(Y_teste, predicao, labels=classes) # cria a matriz de confusão a partir das classes corretas (Y), classes preditas e nome das classes

Câncer_verdadeiro = cm[0, 0] # Quadrante 2
Câncer_falso = cm[1, 0] # Quadrante 3
Normal_verdadeiro = cm[1, 1] # Quadrante 3
Normal_falso = cm[0, 1] # Quadrante 1

print("RESULTADOS:")

print("Normal_verdadeiro: {}".format(Normal_verdadeiro))
print("Câncer_verdadeiro: {}".format(Câncer_verdadeiro))
print("Normal_falso: {}".format(Normal_falso))
print("Câncer_falso: {}\n".format(Câncer_falso))

print("Matrix de confusão: ")
print(cm)

# Com função
acuracia = accuracy_score(Y_teste, predicao)
print("\nAcurácia: {}".format(acuracia))

# "Manual" para confirmar o uso correto da função
acuracia2 = (Câncer_verdadeiro + Normal_verdadeiro)/(Câncer_verdadeiro + Normal_verdadeiro + Câncer_falso + Normal_falso)
#print(acuracia2)

sensibilidade = Câncer_verdadeiro/(Câncer_verdadeiro + Normal_falso)
print("Sensibilidade: {}".format(sensibilidade))

especificidade = Normal_verdadeiro/(Normal_verdadeiro + Câncer_falso)
print("Especificidade: {}\n".format(especificidade))

print("Escore F1: ")
f1_score_macro = f1_score(Y_teste, predicao, average='macro')
f1_score_micro = f1_score(Y_teste, predicao, average='micro')
f1_score_weighted = f1_score(Y_teste, predicao, average='weighted')
print("Macro: {} | Micro: {} | Weighted: {}".format(f1_score_macro, f1_score_micro, f1_score_weighted))

kmeans_clustering = KMeans(n_clusters=2, random_state=0).fit(X) # cria um modelo simples de k-means com 2 clusters, obtido dos dados em X
clusters = kmeans_clustering.predict(X_teste)

print("Escore Silhouette:  {}\n".format(silhouette_score(X_teste, kmeans_clustering.fit_predict(X_teste))))

# e. utilize o método k-means para analisar o dataset de estudo (preparado no item 'a')

# 2 Grupos
print("Para 2 grupos:\n{} dentro do cluster 1".format(list(clusters).count(0)))
print("{} dentro do cluster 2".format(list(clusters).count(1)))

# 3 a 4 Grupos
for i in range(3, 5):
    kmeans_clustering = KMeans(n_clusters=i, random_state=0).fit(X)
    clusters = kmeans_clustering.predict(X_teste)
    print("Para {} grupos:".format(i))
    for j in range(i):     
        print("{} dentro do cluster {}".format(list(clusters).count(j), j+1))
