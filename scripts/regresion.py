import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

class RegresionLineal:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.y_real = None
        self.y_est = None
        self.modelo = None
        self.variables = None
        self.logaritmos_X, self.cocientes_X = self.generar_combinaciones()

    def generar_combinaciones(self):
        """
        Genera cocientes entre las combinaciones de las distintas bandas y logarimos (base 10) de estas (columnas).
        """
        X = self.X
        cocientes = {}
        logaritmos = {}
        columnas = X.columns
        
        for i in range(X.shape[1]):
            for j in range(X.shape[1]):
                if j != i:
                    col1, col2 = columnas[i], columnas[j]
                    q = X[col1] / X[col2].replace(0, np.nan)
                    cocientes[f'{col1}/{col2}'] = q # Genera cocientes

        for i in range(X.shape[1]):
            logaritmos[f'log({columnas[i]})'] = np.log10(X[columnas[i]]) # Genera logaritmos

        return pd.DataFrame(logaritmos), pd.DataFrame(cocientes)

    def evaluar_modelo(self, X, y, variables, n_splits=5, random_state=42):
        """
        Evalúa un modelo de regresión usando validación cruzada.
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        puntaje_r2, puntaje_rmse, puntaje_r2_adj = [], [], []
        modelo = None
        
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            if variables:
                modelo = LinearRegression()
                modelo.fit(X_train[variables], y_train)
                y_pred = modelo.predict(X_test[variables])
            else:
                y_pred = np.full_like(y_test, y_train.mean(), dtype=float) # Si todavía no se agregaron variables, devolver la media como predicción
        
            # Cálculo de R² y RMSE
            r2 = r2_score(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            # Cálculo de R² ajustado
            n_fold = len(y_test) # Número de observaciones
            p = len(variables) if variables else 0 # Número de variables predictoras
            if n_fold > p + 1: # Si el número de variables predictoras es igual al número de observaciones, no puede calcularse el R² ajustado
                r2_ajustado = 1 - (1 - r2) * (n_fold - 1) / (n_fold - p - 1)
            else:
                r2_ajustado = np.nan

            # Añado los valores a las listas
            puntaje_r2.append(r2)
            puntaje_rmse.append(rmse)
            puntaje_r2_adj.append(r2_ajustado)

        return {
            'r2': np.mean(puntaje_r2),
            'rmse': np.mean(puntaje_rmse),
            'r2_adj': np.mean(puntaje_r2_adj)
        }

    def stepwise(self, max_vars=5, n_splits=5, random_state=42, log_y = False, log_X = False, cocientes = False):
        """
        Selección paso a paso (stepwise) usando validación cruzada.
        Se elige la mejor variable en cada paso basada en R².
        """
        y = self.y
        X = self.X
        if log_y:
            y = np.log(y)
        if log_X:
            X = pd.concat([X, self.logaritmos_X], axis=1)
        if cocientes:
            X = pd.concat([X, self.cocientes_X], axis=1)

        variables_seleccionadas = [] 
        variables_restantes = list(X.columns) # Todas las variables

        # Lista para guardar los resultados
        resultados = []

        # Paso 0: predicción con la media
        metricas_base = self.evaluar_modelo(X, y, [], n_splits, random_state)
        resultados.append({
        'Paso': 0,
        'Variable': 'Ninguna',
        'R²': round(metricas_base['r2'], 4),
        'ADJ-R²': round(metricas_base['r2_adj'], 4),
        'RMSE': round(metricas_base['rmse'], 4)
    })

        # Paso X: agregar una variable
        for paso in range(1, max_vars + 1): # Empieza desde 1 hasta max_vars + 1 para no repetir el 0 en el print
            mejor_variable = None
            mejores_metricas = None
            mejor_r2 = -np.inf # Para permitir casos donde la correlación sea negativa

            for var in variables_restantes:
                candidatos = variables_seleccionadas + [var]
                metricas = self.evaluar_modelo(X, y, candidatos, n_splits, random_state)

                if metricas['r2'] > mejor_r2:
                    mejor_r2 = metricas['r2']
                    mejores_metricas = metricas
                    mejor_variable = var

            variables_seleccionadas.append(mejor_variable)
            variables_restantes.remove(mejor_variable)

            resultados.append({
            'Paso': paso,
            'Variable': mejor_variable,
            'R²': round(mejores_metricas['r2'], 4),
            'ADJ-R²': round(mejores_metricas['r2_adj'], 4),
            'RMSE': round(mejores_metricas['rmse'], 4)
        })
   
        self.variables = variables_seleccionadas # Esto lo guardo para después armar la tabla de coeficientes
        self.modelo = LinearRegression() # Guardo el modelo final
        self.modelo.fit(X[variables_seleccionadas],y)
        self.y_real = y  # Para armar las gráficas
        self.y_est = self.modelo.predict(X[variables_seleccionadas])
        return pd.DataFrame(resultados)

    def coeficientes(self):
        if self.modelo is None:
            raise ValueError("No se ajustó un modelo. Ejecutar primero 'stepwise()'.")
        
        coef_df = pd.DataFrame({
            'Variable': self.variables,
            'Coeficiente': self.modelo.coef_
        })

        coef_df.loc[len(coef_df)] = ['Ordenada', self.modelo.intercept_]

        return coef_df

    def graficar(self):
        if self.modelo is None:
            raise ValueError("No se ajustó un modelo. Ejecutar primero 'stepwise()'.")
        
        y_real = self.y_real
        y_pred = self.y_est

        r2 = r2_score(self.y_real, self.y_est)
        plt.figure(figsize=(6, 6))
        plt.scatter(x=y_real, y=y_pred, color = "#17A77E")
        plt.plot([y_real.min(), y_real.max()], [y_real.min(), y_real.max()], color='gray', linestyle='--')
        plt.xlabel('Valor real')
        plt.ylabel('Valor estimado')
        plt.title(f'R² = {r2:.4f}')
        plt.grid(True)
        plt.show()