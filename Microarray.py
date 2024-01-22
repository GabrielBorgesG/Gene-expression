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
'''
# c. CLASSIFICADOR SVM

clf = svm.SVC() # cria um modelo simples de SVM
clf.fit(X, Y) # treina o SVM nos pares de entrada X e Y
# para predizermos a classe de uma amostra desconhecida a, usamos o comando abaixo, onde a é o vetor com a expressão gênica da amostra
# clf.predict(a)
prediction = clf.predict(X)
print(prediction) # mostra as classes preditas para X pelo modelo em clf. Como esse resultado se compara às classes reais em Y?
# para usar o kernel trick devemos passar o kernel como parâmetro na criação do SVM. Um kernel popular é o rbf (Radial Basis Function)
# rbf_svc = svm.SVC(kernel='rbf')
# rbf_svc.fit(X, Y)

# d, MÉTRICAS

print(confusao)
print(acuracia)
print(sensibilidade)
print(especificidade)
print(f1_score)

# e. utilize o método k-means para analisar o dataset de estudo (preparado no item 'a')

# 2 Grupos

print(n_amostras_2a)
print(n_amostras_2b)

# 3 Grupos

print(n_amostras_3a)
print(n_amostras_3b)
print(n_amostras_3c)

# 4 Grupos

print(n_amostras_4a)
print(n_amostras_4b)
print(n_amostras_4c)
print(n_amostras_4d)
'''