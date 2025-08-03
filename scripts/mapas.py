import rasterio as rs
import numpy as np
import plotly.express as px

# Leer imagen
ruta = r"./recorte_acolite/2025-06-09.tif"
with rs.open(ruta) as raster:
    B01 = raster.read(1)
    B02 = raster.read(2)  # azul
    B03 = raster.read(3)  # verde
    B04 = raster.read(4)  # rojo
    B05 = raster.read(5)
    B06 = raster.read(6)
    B07 = raster.read(7)
    B08 = raster.read(8)
    B11 = raster.read(10)

# Calcular NDWI y máscara
ndwi = (B03 - B11) / (B03 + B11 + 1e-6)
ndwi_mask = np.where(ndwi >= 0.6, 1, 0)

# Calcular SDD estimada
sdd = np.exp(-2.42 * (B05 / B03) + 2.09 * (B08 / B06) + 4.15)
mapa_sdd = np.where(ndwi_mask == 1, sdd, np.nan)

# Calcular percentiles para mejorar contraste visual
p5, p95 = np.nanpercentile(mapa_sdd, [5, 95])

# Reemplazar NaN por None (plotly los ignora en hover)
mapa_sdd_clean = np.where(np.isnan(mapa_sdd), None, mapa_sdd)

# Visualizar con Plotly
fig = px.imshow(
    mapa_sdd_clean,
    color_continuous_scale="Rainbow",
    zmin=p5,
    zmax=p95,
    labels={'z': 'Profundidad de disco (cm)'},
    title="Profundidad estimada - 2025-06-09"
)

# Personalizar etiqueta hover (solo z) con estilo blanco
fig.update_traces(
    hovertemplate="<b>%{z:.2f} cm</b><extra></extra>",
    hoverlabel=dict(bgcolor="white", font_size=13, font_color="black")
)

# Ocultar ejes
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)

fig.update_layout(
    coloraxis_colorbar=dict(title="SDD (cm)"),
    margin=dict(l=10, r=10, t=40, b=10)
)

fig.show()
