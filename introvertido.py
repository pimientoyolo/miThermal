# %%
# import mitsuba
import mitsuba as mi, math

# import drjit
import drjit as dr

import numpy as np
import matplotlib.pyplot as plt

# colocar el valor de spectral
mi.set_variant('cuda_ad_spectral')

from mitsuba import ScalarTransform4f as T

# %%
def gausian(lambdas: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    Calculate the Gaussian function value at x with mean mu and standard deviation sigma.
    Args:
        lambdas (np.ndarray): The input values (wavelengths).
        mu (float): The mean of the Gaussian.
        sigma (float): The standard deviation of the Gaussian.
    """
    return np.exp(-0.5 * ((lambdas - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

# %%
def camera_response_gaussian(wavelength: int, sigma:float, k: float = 3, n_wavelents: int = 5 ):
    """
    Camera response function

    Args:
        wavelength (int): Wavelength in nm.
        sigma (float): Standard deviation of the Gaussian.
        k (float, optional): Number of standard deviations. Default is 3.
        n_wavelents (int, optional): Number of wavelengths. Default is 5.
    """

    # define the wavelength range (n values)
    wavelengths = np.linspace(wavelength - k * sigma, wavelength + k * sigma, n_wavelents)

    # define the Gaussian function
    gaussian_values = gausian(wavelengths, wavelength, sigma)

    # normalize the Gaussian function (0,1)
    gaussian_values = (gaussian_values - np.min(gaussian_values)) / (np.max(gaussian_values) - np.min(gaussian_values))

    return wavelengths, gaussian_values
    

# %%
def load_sensor_gaussian(r: float, phi: float, theta: float, wave_lengths: np.ndarray, sigma: float, k: float = 3, n_wavelents: int = 5):
    """
    Load a sensor with Gaussian response.

    Args:
        r (float): Radius of the sensor.
        phi (float): Azimuthal angle in degrees.
        theta (float): Polar angle in degrees.
        wave_lengths (np.ndarray): Wavelengths for the Gaussian response.
        sigma (float): Standard deviation of the Gaussian.
        k (float, optional): Number of standard deviations. Default is 3.
        n_wavelents (int, optional): Number of wavelengths. Default is 5.
    """
    z = r*np.sin(math.radians(theta))
    y = r*np.cos(math.radians(theta))*np.sin(math.radians(phi))
    x = r*np.cos(math.radians(theta))*np.cos(math.radians(phi))
    
    origin = np.array([x, y, z])
    film_dic = {
        'type': 'specfilm',
        'width': 256,
        'height': 256,
        'rfilter': {
            'type': 'tent',
        }
    }

    # for wavelength in wave_lengths:
    #     # Calculate the Gaussian response for the given wavelength
    #     wavelengths, gaussian_values = camera_response_gaussian(wavelength, sigma, k, n_wavelents)

    #     # Convert numpy arrays to comma-separated strings
    #     wavelengths_str = ', '.join(str(int(w)) for w in wavelengths)
    #     gaussian_values_str = ', '.join(f'{v:.6f}' for v in gaussian_values)
        
    #     # Create a sensor with Gaussian response
    #     film_dic[f'band_{wavelength}'] = {
    #         'type': 'irregular',
    #         'wavelengths': wavelengths_str,
    #         'values': gaussian_values_str,
    #     }

    # create sensor uniform
    distancia =  wave_lengths[1] - wave_lengths[0]
    separar = distancia // 2

    for wave_length in wave_lengths:
        
        w_min = int(wave_length - separar)
        w_max = int(wave_length + separar)

        values = '1.0, 1.0'

        # if w_min > 12000:
        #     values = '1.0, 1.0'

        film_dic[f'band_{wave_length}'] = {
            'type': 'regular',
            'wavelength_min': w_min,
            'wavelength_max': w_max,
            'values' : values,
        }



    
    return mi.load_dict({
        'type': 'perspective',
        'fov': 40,
        'to_world': T().look_at(
            origin=origin,
            target=[0, 0, 0],
            up=[0, 0, 1]
        ),
        'sampler': {
            'type': 'independent',
            'sample_count': 1
        },
        'film': film_dic,
        },
    )

# %%
wave_lengths = np.linspace(8000, 14000, 49).astype(int)

sigma = 10
r = 35
phi = 0
theta = 0
k=3
n=15

sensor = load_sensor_gaussian(r, phi, theta, wave_lengths, sigma, k, n)

# %%
dragon = {
    'type': 'obj',
    'filename': 'scenes/objects/dragon.obj',
    'to_world': T().translate([0, 0, 0]),
    'bsdf': {
        'type': 'diffuse',
        'reflectance': {
            'type': 'rgb',
            'value': [0.1, 0.2, 0.3]
        }
    },
    # 'emitter': {
    #     'type': 'area',
    #     'radiance': {
    #         'type': 'regular',
    #         'wavelength_min': 8000,
    #         'wavelength_max': 14000,
    #         'values': '0.1, 0.9'
    #     }
    # }
    'emitter': {
        'type': 'area',
        'radiance': {
            'type': 'blackbody',
            'wavelength_min': 6000,
            'wavelength_max': 16000,
            'temperature': 5000
        },
    }
}

# %%
scene = mi.load_dict({
    'type': 'scene',
    'integrator': {
        'type': 'path',
    },
    'emitter': {
        'type' : 'constant'
    },
    'dragon': dragon,
})

# %%
imagen = mi.render(scene, sensor=sensor, spp=1024*16)

# %%
print(imagen.shape)
print(np.min(imagen))
print(np.max(imagen))

# %%
banda = 0

print(np.min(imagen[:, :, banda]))
print(np.max(imagen[:, :, banda]))

plt.figure(figsize=(10, 10))
plt.imshow(imagen[:, :, banda], cmap='gray')
plt.colorbar()
plt.title('Imagen renderizada')
plt.show()

# %%
scenea = mi.traverse(scene)
print(scenea)

# %%
print(scenea['emitter.radiance.values'])

# %% [markdown]
# ## Visualizar el espectro de un pixel

# %%
x = 200
y = 250

pixel = imagen[y, x, :].numpy()

plt.figure(figsize=(10, 10))
plt.plot(wave_lengths, pixel)
plt.title('Pixel values')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Pixel value')
plt.grid()
plt.show()

# %%
from scipy import constants as const

def blackbody_radiance_nm(wavelengths_nm, temperature):
    """
    Compute spectral radiance B(λ, T) of a black body
    using scipy constants.

    Args:
        wavelengths_nm: array-like of wavelengths in nanometers (nm).
        temperature:    temperature in Kelvin (K).
    
    Returns:
        numpy array of spectral radiance in W·sr⁻¹·m⁻²·nm⁻¹.
    """
    # Convert wavelengths to meters
    wavelengths_m = np.array(wavelengths_nm, dtype=float) * 1e-9
    
    # Planck's law for spectral radiance per meter: W·sr⁻¹·m⁻²·m⁻¹
    B_m = (2 * const.h * const.c**2) / (wavelengths_m**5) / (
        np.exp(const.h * const.c / (wavelengths_m * const.k * temperature)) - 1
    )
    
    # Convert from per meter to per nanometer: 1 m = 1e9 nm
    B_nm = B_m * 1e-9
    
    return B_nm

radiance = blackbody_radiance_nm(wave_lengths, 5000)

# %%
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.plot(wave_lengths, radiance)
plt.title('Blackbody Radiance')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Radiance (W·sr⁻¹·m⁻²·nm⁻¹)')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(wave_lengths, pixel)
plt.title('Pixel values')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Pixel value')
plt.grid()

# %% [markdown]
# ### Visual test SPP

# %%
base = 8
spps = [2**(base), 2**(base+2), 2**(base+4), 2**(base+6)]
imagenes = []

for spp in spps:
    imagen = mi.render(scene, sensor=sensor, spp=spp)
    imagenes.append(imagen)

# %%
banda = 0

plt.figure(figsize=(10, 10))
for i, spp in enumerate(spps):

    plt.subplot(2, 2, i + 1)
    plt.imshow(imagenes[i][:, :, banda], cmap='gray')
    plt.axis('off')
    plt.title(f'Imagen renderizada con {spps[i]} spp')

plt.show()

# %% [markdown]
# ### Grafica del cuerpo negro vs medisiones

# %%
x = 200
y = 250

plt.figure(figsize=(15, 30))
for i in range(len(spps)):

    plt.subplot(4, 2, i*2+1)
    plt.plot(wave_lengths, radiance)
    plt.title('Blackbody Radiance')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Radiance (W·sr⁻¹·m⁻²·nm⁻¹)')
    plt.grid()

    pixel = imagenes[i][y, x, :].numpy()

    plt.subplot(4, 2, i*2+2)
    plt.plot(wave_lengths, pixel)
    plt.title(f'Pixel values with {spps[i]} spp')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Pixel value')
    plt.grid()

plt.show()

# %% [markdown]
# # Idea para asignar una emisiviadad y cuerpo negro

# %% [markdown]
# ## Cargar datos de emisividad

# %%
dataBaseName = np.load('../data/matName_FullDatabase.npy', allow_pickle=True).item()["matName"]
dataBaseName = dataBaseName.squeeze() 
dataBaseName = np.hstack(dataBaseName) # lista de nombres de los materiales

dataBaseLib = np.load('../data/matLib_FullDatabase.npy', allow_pickle=True).item()["matLib"] 
dataBaseLib = dataBaseLib[::-1, :] # Reverso el orden de la base de datos

# vamos a asumir que las 49 firmas espectrales coinciden con las 49 longitudes de onda de la escena

print(dataBaseName.shape)
print(dataBaseLib.shape)



# %%
print(dataBaseName)

# %%
material = "stone" # escoge el material que quieres
indice_material = np.where(dataBaseName == material)[0][0] # busca el indice del material en la base de datos

firma = dataBaseLib[:, indice_material] # firma espectral del material

print(firma.shape)

# plot
plt.figure(figsize=(10, 10))
plt.plot(wave_lengths, firma)
plt.title(f'Firma espectral del material {material}')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Firma espectral')
plt.ylim(0, 1)
plt.grid()
plt.show()

# %% [markdown]
# ### Definir blackbody con la emisividad

# %%
temperatura = 5000 # temperatura objeto

black_body = blackbody_radiance_nm(wave_lengths, temperatura) # radiancia del objeto

emision = black_body * firma # emision del objeto

# subplots
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.plot(wave_lengths, black_body)
plt.title('Blackbody Radiance')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Radiance (W·sr⁻¹·m⁻²·nm⁻¹)')
plt.grid()

plt.subplot(1, 3, 2)
plt.plot(wave_lengths, firma)
plt.title(f'Firma espectral del material {material}')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Firma espectral')
plt.ylim(0, 1)
plt.grid()

plt.subplot(1, 3, 3)
plt.plot(wave_lengths, emision)
plt.title(f'Emision del material {material}')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Emision (W·sr⁻¹·m⁻²·nm⁻¹)')
plt.grid()

plt.show()


# %% [markdown]
# ### Creacion del objeto con esa emision en especifico

# %%
def lista_a_string(valores):

    return ", ".join(str(v) for v in valores)

# %%
dragon = {
    'type': 'obj',
    'filename': 'scenes/objects/dragon.obj',
    'to_world': T().translate([0, 0, 0]),
    'bsdf': {
        'type': 'diffuse',
        'reflectance': {
            'type': 'rgb',
            'value': [0.1, 0.2, 0.3]
        }
    },
    'emitter': {
        'type': 'area',
        'radiance': {
            'type': 'irregular',
            'wavelengths': lista_a_string(wave_lengths),
            'values': lista_a_string(emision),
        }
    }
}

# %% [markdown]
# ### Creacion de sensor (el mismo anterior)

# %%
sigma = 10
r = 35
phi = 0
theta = 0
k=3
n=15

sensor = load_sensor_gaussian(r, phi, theta, wave_lengths, sigma, k, n)

# %% [markdown]
# ### Creaciond de la escena

# %%
scene = mi.load_dict({
    'type': 'scene',
    'integrator': {
        'type': 'path',
    },
    'emitter': {
        'type' : 'constant'
    },
    'dragon': dragon,
})

# %% [markdown]
# ### Renderizacion

# %%
imagen = mi.render(scene, sensor=sensor, spp=1024*16)

# %%
print(imagen.shape)
print(np.min(imagen))
print(np.max(imagen))

# %%
banda = 0

print(np.min(imagen[:, :, banda]))
print(np.max(imagen[:, :, banda]))

plt.figure(figsize=(10, 10))
plt.imshow(imagen[:, :, banda], cmap='gray')
plt.colorbar()
plt.title('Imagen renderizada')
plt.show()

# %% [markdown]
# ### Visualizar firma

# %%
x = 200
y = 250

pixel = imagen[y, x, :].numpy()

plt.figure(figsize=(15, 8))

plt.subplot(1, 2, 1)
plt.plot(wave_lengths, emision)
plt.title(f'Emision del material {material}')
plt.xlabel('Wavelength (nm)')


plt.ylabel('Emision (W·sr⁻¹·m⁻²·nm⁻¹)')
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(wave_lengths, pixel)
plt.title('Pixel values')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Pixel value')
plt.grid()

plt.show()


