import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# guardado y carga de modelos
import joblib

class RegresionML:
    def __init__(self, X, y, ratios = False):
        
        self.X = X
        self.y = y
        self.ratios = ratios

        # espacio para modelos y parámetros
        self.rf_mejor_modelo = None
        self.rf_mejores_parametros = None
        self.xgb_mejor_modelo = None
        self.xgb_mejores_parametros = None

        cocientes = {}
        columnas = X.columns
        
        if ratios == True:
            for i in range(X.shape[1]):
                for j in range(X.shape[1]):
                    if j != i:
                        col1, col2 = columnas[i], columnas[j]
                        q = X[col1] / X[col2].replace(0, np.nan)
                        cocientes[f'{col1}/{col2}'] = q # Genera cocientes
            
            self.X = pd.concat([self.X, pd.DataFrame(cocientes)], axis=1)
    
        # división de datos
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X,self.y, test_size = 0.2, random_state = 42)

    def cargar_modelo(self, model = "rf", filename = None):
        if model == "rf":
            self.rf_mejor_modelo = joblib.load(f"..//modelos_ml//{filename}")
        elif model == "xgb":
            self.xgb_mejor_modelo = joblib.load(f"..//modelos_ml//{filename}")
        else:
            raise ValueError("Especificar nombre del modelo")

    def train_rf(self):
        param_grid = {
            "n_estimators": [200, 300, 5,00],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt", "log2"],
            "bootstrap": [True, False]
        }

        grid_search = GridSearchCV(
            estimator = RandomForestRegressor(random_state = 42, n_jobs = -1),
            param_grid = param_grid,
            cv = 3,
            scoring = "r2",
            verbose = 2
        )
        

        grid_search.fit(self.X_train, self.y_train)
        self.rf_mejor_modelo = grid_search.best_estimator_
        self.rf_mejores_parametros = grid_search.best_params_

        y_pred = self.rf_mejor_modelo.predict(self.X_test)
        y_pred_global = self.rf_mejor_modelo.predict(self.X)
        self.mae = round(mean_absolute_error(self.y_test, y_pred), 0)
        self.rmse = round(mean_squared_error(self.y_test, y_pred), 0)
        self.r2 = round(r2_score(self.y_test, y_pred), 4)
        self.r2_global = round(r2_score(y_pred_global, self.y), 4)


        joblib.dump(grid_search.best_estimator_, f"..//modelos_ml//rf_{datetime.now().strftime('%y%m%d')}_{self.r2_global}_{self.r2}.pkl") 

        return {"MAE": self.mae, "RMSE": self.rmse, "R2": self.r2}

    def train_xgb(self):
        param_grid = {
            "n_estimators": [200, 300, 500],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.2],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "gamma": [0, 1],
            "reg_alpha": [0, 0.1],
            "reg_lambda": [1, 2]
            }

        grid_search = GridSearchCV(
            estimator=XGBRegressor(
            random_state=42,
            n_jobs=-1,
            verbosity=0),
            param_grid=param_grid,
            cv=3,
            scoring="r2",
            n_jobs=-1,
            verbose=2
        )
        grid_search.fit(self.X_train, self.y_train)

        self.xgb_mejor_modelo = grid_search.best_estimator_
        self.xgb_mejores_parametros = grid_search.best_params_

        y_pred = self.xgb_mejor_modelo.predict(self.X_test)
        y_pred_global = self.xgb_mejor_modelo.predict(self.X)
        self.mae = round(mean_absolute_error(self.y_test, y_pred), 2)
        self.rmse = round(mean_squared_error(self.y_test, y_pred), 2)
        self.r2 = round(r2_score(self.y_test, y_pred), 4)
        self.r2_global = round(r2_score(y_pred_global, self.y), 4)

        joblib.dump(grid_search.best_estimator_, f"..//modelos_ml//xgb_{datetime.now().strftime('%y%m%d')}_{self.r2_global}_{self.r2}.pkl") 

        return {"MAE": self.mae, "RMSE": self.rmse, "R2": self.r2}
    
    def graficar(self, model = "rf"):
        ratios = self.ratios
        if model == "rf":
            modelo = self.rf_mejor_modelo
        elif model == "xgb":
            modelo = self.xgb_mejor_modelo
        else:
            raise ValueError("Modelo debe ser 'rf' o 'xgb'")
        
        if modelo is None:
            raise ValueError("No se entrenó el modelo solicitado")
        
        y_pred = modelo.predict(self.X)
        m, b = np.polyfit(self.y, y_pred, 1)
        y_tendencia = m * self.y + b

        y_train_pred = modelo.predict(self.X_train)
        y_test_pred = modelo.predict(self.X_test)
        
        fig, ax = plt.subplots(figsize=(5, 5))

        r2_global = round(r2_score(y_pred, self.y), 4)
        rmse_global = round(mean_squared_error(y_pred, self.y), 0)


        ax.scatter(self.y_train, y_train_pred, color="#17A77E", alpha = 0.3, label="Datos de entrenamiento")
        ax.scatter(self.y_test, y_test_pred, color="#fc6f03", alpha = 0.3, label="Datos de testeo", marker='^')
        ax.plot(self.y, y_tendencia, color="#117766", label="Línea de tendencia")
        ax.plot([self.y.min(), self.y.max()], [self.y.min(), self.y.max()], linestyle='--', color='gray', label="Ideal")
        ax.set_xlabel('SDD real (cm)')
        ax.set_ylabel('SDD estimada (cm)')
        ax.text(0.5,0.05,f'R²: {r2_global}\nR²-testeo: {self.r2}\nRMSE: {rmse_global} (cm)\nRMSE-testeo: {self.rmse} (cm)', transform=ax.transAxes, bbox=dict(boxstyle="round, pad=0.5", facecolor="white", edgecolor="gray", alpha=0.6))
        ax.legend()
        ax.axis("equal")
        ax.grid(True)
        if model == "rf":
            if ratios == True:
                ax.set_title("Random Forest (Cocientes incorporados)")
                plt.savefig(f"..//modelos_ml//rf_ratios_{datetime.now().strftime('%y%m%d')}_{r2_global}_{self.r2}.png", dpi=300, bbox_inches='tight')
            else:
                ax.set_title("Random Forest")
                plt.savefig(f"..//modelos_ml//rf_{datetime.now().strftime('%y%m%d')}_{r2_global}_{self.r2}.png", dpi=300, bbox_inches='tight')
        elif model == "xgb":
            if ratios == True:
                ax.set_title("XGBoost (Cocientes incorporados)")
                plt.savefig(f"..//modelos_ml//xgb_ratios_{datetime.now().strftime('%y%m%d')}_{r2_global}_{self.r2}.png", dpi=300, bbox_inches='tight')
            else:
                ax.set_title("XGBoost")
                plt.savefig(f"..//modelos_ml//xgb_{datetime.now().strftime('%y%m%d')}_{r2_global}_{self.r2}.png", dpi=300, bbox_inches='tight')

