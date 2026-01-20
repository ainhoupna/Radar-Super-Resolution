import rasterio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# ------------------- CONFIGURACIÓN -------------------
input_tif = '2023-09-02--07_00_x2.tif'          # TU ÚNICO .TIF CON DOS MAPAS
output_png = 'comparacion_zoom_pamplona_dos_mapasx2_final.png'  # Salida

zoom_color = 'black'  # Bordes negros
rect_width = 4

inset1_pos = [0.08, 0.72, 0.35, 0.35]  # Inset izquierda: zoom mapa original
inset2_pos = [0.57, 0.72, 0.35, 0.35]  # Inset derecha: zoom mapa SR

zoom_factor = 4.0  # Amplificación

# Zona relativa en Pamplona (ajusta si necesitas)
rel_x = 0.35   # fracción horizontal dentro de cada mapa (0.65 ≈ centro-derecha de Pamplona)
rel_y = 0.40   # fracción vertical (centrado en la zona roja)

zoom_width = 350
zoom_height = 350
# ----------------------------------------------------

# Cargar imagen única (dos mapas side-by-side)
with rasterio.open(input_tif) as src:
    img = src.read()
    if img.ndim == 3:
        img = np.moveaxis(img, 0, -1)
    height, width = img.shape[:2]

print(f"Dimensiones totales: {width}x{height}")

# Asumir mitad izquierda = mapa original, mitad derecha = mapa SR
half_width = width // 2
img_left = img[:, 0:half_width, :]   # Mapa izquierda (baja res)
img_right = img[:, half_width:width, :]  # Mapa derecha (alta res)

left_width = half_width
right_width = width - half_width

# Calcular coordenadas relativas para la misma zona geográfica
zoom1_x = int(left_width * rel_x)
zoom1_y = int(height * rel_y)
zoom1_x, zoom1_y, zoom1_width, zoom1_height = max(0, zoom1_x), max(0, zoom1_y), zoom_width, zoom_height

zoom2_x = int(right_width * rel_x) + half_width  # offset para la derecha
zoom2_y = int(height * rel_y)
zoom2_x, zoom2_y, zoom2_width, zoom2_height = max(0, zoom2_x), max(0, zoom2_y), zoom_width, zoom_height

# Validación (evitar salidas)
def validate(x, y, w, h, max_w, max_h):
    x = max(0, min(x, max_w - w))
    y = max(0, min(y, max_h - h))
    w = min(w, max_w - x)
    h = min(h, max_h - y)
    return x, y, w, h

zoom1_x, zoom1_y, zoom1_width, zoom1_height = validate(zoom1_x, zoom1_y, zoom1_width, zoom1_height, width, height)
zoom2_x, zoom2_y, zoom2_width, zoom2_height = validate(zoom2_x, zoom2_y, zoom2_width, zoom2_height, width, height)

# Recortes
zoom1_crop = img[zoom1_y:zoom1_y + zoom1_height, zoom1_x:zoom1_x + zoom1_width]
zoom2_crop = img[zoom2_y:zoom2_y + zoom2_height, zoom2_x:zoom2_x + zoom2_width]

# Figura
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111)
ax.imshow(img)
ax.axis('off')

# Rectángulos negros en cada mapa
rect1 = Rectangle((zoom1_x, zoom1_y), zoom1_width, zoom1_height, linewidth=rect_width, edgecolor=zoom_color, facecolor='none')
rect2 = Rectangle((zoom2_x, zoom2_y), zoom2_width, zoom2_height, linewidth=rect_width, edgecolor=zoom_color, facecolor='none')
ax.add_patch(rect1)
ax.add_patch(rect2)

# Inset 1: Zoom del mapa izquierda (original)
inset_ax1 = fig.add_axes(inset1_pos)
inset_ax1.imshow(zoom1_crop)
inset_ax1.axis('off')
inset_ax1.add_patch(Rectangle((0, 0), zoom1_width * zoom_factor, zoom1_height * zoom_factor, linewidth=rect_width, edgecolor=zoom_color, facecolor='none'))
inset_ax1.set_title('Original Map', color=zoom_color, fontsize=14, pad=12)

# Inset 2: Zoom del mapa derecha (SR)
inset_ax2 = fig.add_axes(inset2_pos)
inset_ax2.imshow(zoom2_crop)
inset_ax2.axis('off')
inset_ax2.add_patch(Rectangle((0, 0), zoom2_width * zoom_factor, zoom2_height * zoom_factor, linewidth=rect_width, edgecolor=zoom_color, facecolor='none'))
inset_ax2.set_title('Super-Res Map x2', color=zoom_color, fontsize=14, pad=12)

# Guardar
plt.subplots_adjust(left=0, right=1, top=0.98, bottom=0.02)
plt.savefig(output_png, dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()

print(f"¡Guardada como {output_png}!")
print("Dos zooms en la misma zona de Pamplona, uno por cada mapa.")