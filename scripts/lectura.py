import pandas as pd
# Rutas a los archivos CSV
archivo_reflectancias = "datos\\base_de_datos_gis.csv"  # contiene: fecha,punto,pixel,banda,reflect,longitud,latitud
archivo_parametros = "datos\\base_de_datos_lab.csv"     # contiene: fecha,longitud,latitud,param,valor

# Leer los archivos
df_reflect = pd.read_csv(archivo_reflectancias)
df_param = pd.read_csv(archivo_parametros)

# Filtrar los parámetros "secchi"
df_secchi = df_param[df_param["param"].str.lower() == "secchi"]

# Merge por fecha y coordenadas
merged = pd.merge(
    df_secchi,
    df_reflect,
    on=["fecha", "latitud", "longitud"],
    how="inner"
)

# Pivotear la tabla para poner bandas como columnas
tabla_final = merged.pivot_table(
    index=["param", "fecha", "longitud", "latitud", "valor"], 
    columns="banda",
    values="reflect"
).reset_index()

# Reordenar columnas: param | B01 | B02 | ... | B8A
bandas = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B11', 'B12', 'B8A']
columnas_finales = ['valor'] + bandas

# Crear Tabla final
df = tabla_final[columnas_finales]