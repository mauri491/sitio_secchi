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
        self.X_extendido = pd.concat([X, self.generar_combinaciones()], axis=1)
        self.modelo_final = None
        self.variables_seleccionadas = [] 
        
    def generar_combinaciones(self):
        """
        Genera cocientes entre las combinaciones de las distintas bandas (columnas).
        """
        X = self.X
        cocientes = {}
        columnas = X.columns
        for i in range(X.shape[1]):
            for j in range(X.shape[1]):
                if j != i:
                    col1, col2 = columnas[i], columnas[j]
                    q = X[col1] / X[col2].replace(0, np.nan)
                    cocientes[f'{col1}/{col2}'] = q
        return pd.DataFrame(cocientes)

    def evaluar_modelo(self, X, y, variables, n_splits=5, random_state=42):
        """
        Evalúa un modelo de regresión usando validación cruzada.
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        r2_scores, rmse_scores = [], []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            if variables:
                modelo = LinearRegression()
                modelo.fit(X_train[variables], y_train)
                y_pred = modelo.predict(X_test[variables])
            else:
                y_pred = np.full_like(y_test, y_train.mean(), dtype=float)

            r2_scores.append(r2_score(y_test, y_pred))
            rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))

        return {
            'r2': np.mean(r2_scores),
            'rmse': np.mean(rmse_scores)
        }

    def stepwise(self, max_vars=5, n_splits=5, random_state=42, log=False, extendido=False):
        """
        Selección paso a paso (stepwise) usando validación cruzada.
        Se elige la mejor variable en cada paso basada en R².
        """
        y = np.log10(self.y) if log else self.y
        X = self.X_extendido if extendido else self.X

        variables_seleccionadas = []
        variables_restantes = list(X.columns)

        # Paso 0: predicción con la media
        metricas_base = self.evaluar_modelo(X, y, [], n_splits, random_state)
        print(f"Paso 0: Solo la media  — R²: {metricas_base['r2']:.4f}, RMSE: {metricas_base['rmse']:.4f}")

        for paso in range(1, max_vars + 1):
            mejor_variable = None
            mejores_metricas = None
            mejor_r2 = -np.inf

            for var in variables_restantes:
                candidatos = variables_seleccionadas + [var]
                metricas = self.evaluar_modelo(X, y, candidatos, n_splits, random_state)

                if metricas['r2'] > mejor_r2:
                    mejor_r2 = metricas['r2']
                    mejores_metricas = metricas
                    mejor_variable = var

            variables_seleccionadas.append(mejor_variable)
            variables_restantes.remove(mejor_variable)

            print(f"Paso {paso}: Se añade '{mejor_variable}' — R²: {mejores_metricas['r2']:.4f}, RMSE: {mejores_metricas['rmse']:.4f}")

        # Guardo las variables seleccionadas
        self.variables_seleccionadas = variables_seleccionadas

    def ajustar_modelo_final(self, log=False, extendido=False):
        """
        Ajusta el modelo final con las variables seleccionadas y guarda el modelo.
        """
        if not self.variables_seleccionadas:
            raise ValueError("No hay variables seleccionadas. Ejecutar primero 'stepwise()'.")

        y = np.log10(self.y) if log else self.y
        X = self.X_extendido if extendido else self.X

        modelo = LinearRegression()
        modelo.fit(X[self.variables_seleccionadas], y)

        self.modelo_final = modelo  # Guarda el modelo
        coeficientes = pd.Series(modelo.coef_, index=self.variables_seleccionadas)
        intercepto = modelo.intercept_

        print("\nModelo final ajustado:")
        print("Coeficientes:")
        print(coeficientes.to_string())
        print(f"Intercepto: {intercepto:.4f}")

    def graficar(self, log=False, extendido=False):
        if self.modelo_final is None:
            raise ValueError("No se ajustó un modelo. Ejecutar primero 'stepwise()' y 'ajustar_modelo_final()'.")

        # Obtener X e y según flags
        X = self.X_extendido if extendido else self.X
        y_real = np.log10(self.y) if log else self.y

        # Predecir
        y_pred = self.modelo_final.predict(X[self.variables_seleccionadas])

        # Calcular R²
        r2 = r2_score(y_real, y_pred)

        # Gráfico
        plt.figure(figsize=(8, 6))
        plt.scatter(x=y_real, y=y_pred, color = "#17A77E")
        plt.plot([y_real.min(), y_real.max()], [y_real.min(), y_real.max()], color='gray', linestyle='--', label='Ideal (y = ŷ)')
        plt.xlabel('Valor real')
        plt.ylabel('Predicción')
        plt.title(f'Predicción vs Valor real — R² = {r2:.4f}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


