#!/usr/bin/env python3

import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt
import pickle

mi.set_variant("cuda_ad_rgb")
from mitsuba import ScalarTransform4f as T

# Parámetros del medio
sigma_t_value = 0.001  # Menor densidad para permitir ver a través
albedo_value = 0.0     # No scattering

# Medio homogéneo
medium_dict = {
    "type": "homogeneous",
    "id": "volumetric_medium",
    "sigma_t": {
        "type": "rgb",
        "value": [sigma_t_value] * 3
    },
    "albedo": {
        "type": "rgb",
        "value": [albedo_value] * 3
    }
}

# Cubo principal
cube_dict = {
    "type": "cube",
    "to_world": T().scale([2.0, 2.0, 2.0]),
    "bsdf": {
        "type": "diffuse",
        "reflectance": {"type": "rgb", "value": [0.8, 0.4, 0.2]}
    }
}

# Esfera azul al fondo
background_sphere = {
    "type": "sphere",
    "center": [0, 0, -20],
    "radius": 4.0,
    "bsdf": {
        "type": "diffuse",
        "reflectance": {"type": "rgb", "value": [0.2, 0.6, 0.9]}
    }
}

# Emisor de ambiente
environment_emitter = {
    "type": "constant",
    "radiance": {
        "type": "rgb",
        "value": [1.0, 1.0, 1.0]
    }
}

# Sensor
sensor_dict = {
    "type": "perspective",
    "fov": 45,
    "to_world": T().look_at(
        origin=[0, -20, 0],
        target=[0, 0, 0],
        up=[0, 0, 1]
    ),
    "film": {
        "type": "hdrfilm",
        "width": 512,
        "height": 512,
        "rfilter": {"type": "box"}
    },
    "sampler": {"type": "independent", "sample_count": 256},
    "medium": {"type": "ref", "id": "volumetric_medium"}
}

integrator_dict = {"type": "volpath"}

# Escena
scene_dict = {
    "type": "scene",
    "integrator": integrator_dict,
    "medium": medium_dict,
    "cube": cube_dict,
    "background": background_sphere,
    "emitter": environment_emitter,
    "sensor": sensor_dict
}

scene = mi.load_dict(scene_dict)

print("Renderizando escena con medio volumétrico...")
image = mi.render(scene)

if hasattr(image, 'numpy'):
    image_array = image.numpy()
else:
    bitmap = mi.util.convert_to_bitmap(image)
    image_array = np.array(bitmap)

plt.figure(figsize=(8, 8))
plt.imshow(np.clip(image_array, 0.0, 1.0))
plt.axis("off")
plt.title("Escena con niebla y fondo visible")
plt.savefig("output/volumetric_medium_with_background.png", dpi=150, bbox_inches="tight")
plt.show()

print("Guardando resultados...")
with open("output/volumetric_medium_results.pkl", "wb") as f:
    pickle.dump({
        "sigma_t_value": sigma_t_value
    }, f)

print("¡Listo! Imagen guardada en 'output/volumetric_medium_with_background.png'")
