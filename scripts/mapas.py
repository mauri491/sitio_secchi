import warnings
import numpy as np
import rasterio as rs
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

warnings.filterwarnings("ignore")

def generar_mapa(fecha):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
    with rs.open(f".\\recorte_acolite\\{fecha}.tif") as raster:
        bandas = { 
            'B01': raster.read(1), 'B02': raster.read(2), 'B03': raster.read(3),
            'B04': raster.read(4), 'B05': raster.read(5), 'B06': raster.read(6),
            'B07': raster.read(7), 'B08': raster.read(8), 'B8A': raster.read(9),
            'B11': raster.read(10), 'B12': raster.read(11)
        }
        meta = raster.meta.copy()  

    def stretch(banda):
        p2, p98 = np.percentile(banda, (2, 98))
        return np.clip((banda - p2) / (p98 - p2 + 1e-6), 0, 1)

    # RGB 
    rgb = np.stack([stretch(bandas['B04']), stretch(bandas['B03']), stretch(bandas['B02'])], axis=-1)

    # NDWI 
    ndwi = (bandas['B03'] - bandas['B11']) / (bandas['B03'] + bandas['B11'])
    ndwi_vals = ndwi[np.isfinite(ndwi)]
    otsu_val = threshold_otsu(ndwi_vals)

    if np.abs(otsu_val) > 1:
        threshold = np.mean(ndwi_vals)
    else:
        threshold = otsu_val

    ndwi_mask = ndwi >= threshold

    # estimación de turbidez
    ec = np.exp(-2.42*(bandas['B05']/bandas['B03']) + 2.09*(bandas['B08']/bandas['B06']) + 4.15)
    turb = np.where(ndwi_mask == 1, ec, np.nan)
    p5, p95 = np.nanpercentile(turb, [5, 95])

    # actualizar metadatos
    meta.update({
        "count": 1,
        "dtype": "float32",
        "nodata": np.nan
    })

    # guardar raster de turbidez como GeoTIFF
    output_file = f".//mapas//turb_{fecha}.tif"
    with rs.open(output_file, "w", **meta) as dst:
        dst.write(np.clip(turb, p5, p95).astype("float32"), 1)

    # configurar recuadros
    for ax in axs:
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    axs[0].imshow(rgb)
    axs[0].set_title('TrueColor')

    axs[1].imshow(ndwi_mask, cmap='gray')
    axs[1].set_title('NDWI')

    im = axs[2].imshow(turb, cmap='rainbow', vmin=p5, vmax=p95)
    axs[2].set_title('Estimación')

    # barrita de colores
    cbar_ax = fig.add_axes([0.92, 0.16, 0.02, 0.68])
    fig.colorbar(im, cax=cbar_ax, label="SDD (cm)")

    fig.tight_layout(rect=[0, 0, 0.9, 1])

    plt.show()
