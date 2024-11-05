import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Função de construção da lista de pontos classificados incorretamente
def constroiListaPCI(x_bias, y, w):

    '''
    ### 1. Função de Construção da Lista de Pontos Classificados Incorretamente

    A função `constroiListaPCI` é responsável por identificar quais pontos no conjunto de dados são classificados incorretamente pelo modelo atual do Perceptron. Ela realiza as seguintes etapas:

    - Cálculo das Predições: Utiliza o produto escalar entre os dados com o bias (`x_bias`) e os pesos (`w`) para gerar as predições.
    - Identificação dos Erros: Compara as predições com os rótulos verdadeiros (`y`) e cria uma lista de índices onde as predições não correspondem aos rótulos.
    '''
    
    predictions = np.sign(np.dot(x_bias, w))  # produto escalar
    l = [i for i in range(len(y)) if predictions[i] != y[i]]  # índices dos pontos incorretos
    return l

# Algoritmo de Aprendizagem do Perceptron
def PLA(X, y, max_iter=50000):

    '''
    ### 2. Algoritmo de Aprendizagem do Perceptron (PLA)

    A função `PLA` implementa o algoritmo do Perceptron para treinar o modelo. O processo é o seguinte:

    - Preparação dos Dados: Adiciona um termo de bias aos dados e inicializa os pesos com zeros.
    - Iterações de Treinamento: Em cada iteração, a função `constroiListaPCI` é usada para encontrar os pontos classificados incorretamente. Se não houver erros, o treinamento é concluído.
    - Atualização dos Pesos: Seleciona um ponto de erro aleatoriamente e ajusta os pesos de acordo com o erro encontrado.
    - Critério de Parada: O loop termina quando todos os pontos são classificados corretamente ou quando o número máximo de iterações (`max_iter`) é atingido.
    '''
    
    x_bias = np.c_[np.ones(X.shape[0]), X]  # Adiciona o bias
    w = np.zeros(x_bias.shape[1])  # Inicializa pesos
    it = 0
    
    while True:
        l = constroiListaPCI(x_bias, y, w)
        
        if len(l) == 0:  # Se não houver mais pontos incorretamente classificados
            print(f"Solução encontrada após {it} iterações.")
            break
        
        idx = np.random.choice(l)  # Escolhe um ponto incorretamente classificado aleatoriamente
        x_i = x_bias[idx]
        y_i = y[idx]
        w += y_i * x_i  # Atualiza os pesos
        it += 1
    
        if it == max_iter:
            print("Limite máximo de iterações atingido.")
            break
    
    return it, w
    
def classify(X, w):

    '''
    ### 3. Função de Classificação

    A função `classify` usa o modelo treinado para fazer previsões sobre novos dados. Adiciona o bias aos dados e calcula a predição usando o produto escalar com os pesos.
    '''
    
    x_bias = np.c_[np.ones(X.shape[0]), X]  # Adiciona o bias
    return np.sign(np.dot(x_bias, w))
