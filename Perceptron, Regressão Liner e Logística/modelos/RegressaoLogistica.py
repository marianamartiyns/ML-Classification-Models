import numpy as np
import numpy.linalg as LA

class LogisticRegression:
    def __init__(self, eta=0.001, tmax=1500, epsilon=1e-5):
        self.eta = eta
        self.tmax = tmax
        self.epsilon = epsilon
        self.w = None

    def fit(self, _X, _y):
        '''
        O método `fit` ajusta o modelo aos dados usando gradiente descendente. Durante o treinamento, o modelo ajusta os pesos para minimizar a função de custo.

        A atualização dos pesos é dada por: w  =w - taxa de aprendizado . gradiente da função de custo

        O gradiente é calculado como: - 1 / N.X^T (y - probs)

        onde probs são as probabilidades previstas, calculadas pela função sigmoide: 1 / (1 + e^{-z}) 
        e z é o produto escalar entre os dados e os pesos.
        '''
        
        X = np.c_[np.ones((_X.shape[0], 1)), _X]
        y = np.array(_y)
        N, d = X.shape

        # Inicializa w com zeros
        self.w = np.zeros(X.shape[1])
        
        for t in range(self.tmax):
            # Calcula as probabilidades usando a função sigmoide
            z = np.dot(X, self.w)
            probs = 1 / (1 + np.exp(-z))

            # Calcula o gradiente
            errors = y - probs
            gradient = -np.dot(X.T, errors) / N

            # Atualiza os pesos
            self.w -= self.eta * gradient

            # Verifica a convergência
            if LA.norm(gradient) < self.epsilon:
                break

    def predict_prob(self, X):
        '''
        Este método calcula as probabilidades associadas a cada amostra usando a função sigmoide. A fórmula para calcular a probabilidade é:
            
              1 / (1 + e^{-z}) 
        
        onde z é o produto escalar entre os dados e os pesos.
        '''
    
        X = np.array(X)
        z = np.dot(X, self.w)
        probs = 1 / (1 + np.exp(-z))
        return probs

    def predict(self, X):
        '''
        O método `predict` converte as probabilidades em classes binárias. A classificação é feita da seguinte maneira:

            y_pred = 1, se probs >= 0.5
                   = -1, caso contrário
        '''
        probs = self.predict_prob(X)
        return (probs >= 0.5).astype(int) * 2 - 1

    def getW(self):
        return self.w

    def getRegressionY(self, regressionX, shift=0):
        '''
        Este método calcula a linha de decisão da regressão logística. A fórmula para a linha de decisão é:
        
        y = -w_0 + {shift} - w_1 . X / w_2

        onde w_0, w_1 e w_2 são os pesos do modelo e shift é um valor opcional para ajustar a linha.
        '''
        if self.w is None or len(self.w) != 3:
            raise ValueError("O modelo não está ajustado ou não tem 3 coeficientes.")
        return (-self.w[0] + shift - self.w[1] * regressionX) / self.w[2]
