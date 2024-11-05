import numpy as np

class LinearRegression:
    
    def fit(self, X, y):
        '''
        O método `fit` é responsável pelo treinamento do modelo. Ele realiza as seguintes etapas:

        - Adição do Termo de Bias: Adiciona uma coluna de uns aos dados de entrada `X` para incorporar o termo de bias no modelo.
        - Cálculo dos Pesos: Utiliza a equação normal para calcular os pesos do modelo. A equação é dada por: f{w} = (X_b^T X_b)^{-1} X_b^T y

          onde X_b é a matriz dos dados com o bias incluído, e y são os rótulos.
        '''
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # bias
        y = np.array(y)
        self.w = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    
    def predict(self, X):
        '''
        O método `predict` faz previsões com base nos dados de entrada `X`:

        - Adição do Termo de Bias: Adiciona uma coluna de uns aos dados de entrada `X`, semelhante ao que foi feito no método `fit`.
        - Cálculo das Predições: Calcula as predições usando os pesos aprendidos.
        '''
        X_b = np.c_[np.ones((X.shape[0], 1)), X]  # bias
        return X_b.dot(self.w)
    
    def getW(self):
        '''
        O método `getW` retorna os pesos do modelo que foram aprendidos durante o treinamento. Isso permite verificar quais valores foram ajustados pelo modelo.
        '''
        return self.w
