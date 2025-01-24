"""
Moon: https://svs.gsfc.nasa.gov/4720/
- multiply by 100
Mercury: https://astrogeology.usgs.gov/search/map/mercury_messenger_global_dem_665m
- 10x downsize, take left half, resize, divide by 120000
Venus: https://astrogeology.usgs.gov/search/map/venus_magellan_global_topography_4641m
       https://astrogeology.usgs.gov/search/map/venus_magellan_global_c3_mdir_colorized_topographic_mosaic_6600m
- from matplotlib.colors import import rgb_to_hsv
- divide by 90000, fit hsv to unmasked part using histgradientboostingregressor
Mars: https://astrogeology.usgs.gov/search/map/mars_mgs_mola_dem_463m
- 10x downsize, take left half, resize, divide by 60000
"""
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from scipy.ndimage import binary_dilation, zoom
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from matplotlib.colors import rgb_to_hsv
from matplotlib.cm import get_cmap
from tqdm import tqdm
import os
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


def colour_maps(cmap: str, reverse: bool = False, n: int = 50) -> None:
    colours = np.round(get_cmap(cmap)(np.linspace(0, 1, n))[:, :3] * 255, 3)
    if reverse:
        return np.flip(colours, axis=0).tolist()
    return colours.tolist()


def get_planet() -> np.ndarray:
    # Earth
    path = r'climate\data\wc2.1_5m_elev.tif'
    elevation = np.array(Image.open(path))
    # elevation = cassini_projection(elevation)
    elevation = zoom(elevation, 180 / elevation.shape[0], order=0)
    elevation[elevation == -32768] = 0
    with open('a.txt', 'w') as f:
        f.write(str(elevation.tolist()))
    print(np.min(elevation), np.max(elevation))

    # Moon
    path = r'climate\data\moon.tif'
    elevation = np.array(Image.open(path))
    elevation = 1000 * zoom(elevation, 180 / elevation.shape[0], order=0)
    with open('a.txt', 'w') as f:
        f.write(str(elevation.tolist()))
    print(np.min(elevation), np.max(elevation))

    # Mercury
    path = r'climate\data\mercury.tif'
    elevation = Image.open(path)
    elevation = np.array(elevation.resize((elevation.size[0] // 5, elevation.size[1] // 10), Image.NEAREST))
    elevation = elevation[:, :elevation.shape[1] // 2]
    elevation = zoom(elevation, 180 / elevation.shape[0]) / 120000
    with open('a.txt', 'w') as f:
        f.write(str(elevation.tolist()))
    print(np.min(elevation), np.max(elevation))

    # Mars
    path = r'climate\data\mars.tif'
    elevation = Image.open(path)
    elevation = np.array(elevation.resize((elevation.size[0] // 5, elevation.size[1] // 10), Image.NEAREST))
    elevation = elevation[:, :elevation.shape[1] // 2]
    elevation = zoom(elevation, 180 / elevation.shape[0]) / 60000
    with open('a.txt', 'w') as f:
        f.write(str(elevation.tolist()))
    print(np.min(elevation), np.max(elevation))

    # Venus
    path1 = r'climate\data\venus.tif'
    path2 = r'climate\data\venus2.tif'
    elevation = Image.open(path2)
    elevation = np.array(elevation.resize((elevation.size[0], elevation.size[1] // 2), Image.NEAREST))
    elevation = elevation[:, :elevation.shape[1] // 2] / 90000
    mask = binary_dilation(elevation < -10000)
    colour = rgb_to_hsv(np.array(Image.open(path1).resize((elevation.shape[1], elevation.shape[0]), Image.NEAREST)))
    x = colour[~mask, :]
    y = elevation[~mask]
    model = HistGradientBoostingRegressor(random_state=0).fit(x, y)
    H, W = elevation.shape
    y_full = model.predict(colour.reshape(H * W, 3)).reshape(H, W)
    final = elevation.copy()
    final[mask] = y_full[mask]
    final = zoom(final, 180 / final.shape[0])
    with open('a.txt', 'w') as f:
        f.write(str(final.tolist()))
    print(np.min(final), np.max(final))


def cassini_projection(map: np.ndarray, inverse: bool = False) -> np.ndarray:
    """Apply the Cassini projection to an equirectangular map (or the inverse transformation)."""
    H, W = map.shape
    x, y = np.meshgrid(np.arange(0, 360, 360 / W), np.arange(0, 180, 180 / H))
    longitude = 180 - x - (90 / H)
    latitude = 90 - y - (90 / H)

    if inverse:
        longitude *= -np.pi / 180
        latitude *= np.pi / 180
        longitude_new = np.arctan2(np.tan(latitude), np.cos(longitude)) * (180 / np.pi)
        latitude_new = np.arcsin(np.cos(latitude) * np.sin(longitude)) * (180 / np.pi)
    else:
        longitude *= np.pi / 180
        latitude *= np.pi / 180
        longitude_new = -np.arctan2(np.tan(latitude), np.cos(longitude)) * (180 / np.pi)
        latitude_new = np.arcsin(np.cos(latitude) * np.sin(longitude)) * (180 / np.pi)

    x_new = (180 - longitude_new - (90 / H)) * W / 360
    y_new = (90 - latitude_new - (90 / H)) * H / 180
    return map[np.floor(y_new).astype(int), np.floor(x_new).astype(int)]


def fill_climate(cities: pd.DataFrame) -> pd.DataFrame:
    temperature = []
    precipitation = []
    for i in range(12):
        symbol = f'0{i + 1}' if i + 1 < 10 else str(i + 1)
        path = os.path.join(TEMPERATURE_PATH, f'wc2.1_5m_tavg_{symbol}.tif')
        temperature.append(np.array(Image.open(path), dtype=float))
        path = os.path.join(PRECIPITATION_PATH, f'wc2.1_5m_prec_{symbol}.tif')
        precipitation.append(np.array(Image.open(path), dtype=float))
    temperature = np.stack(temperature)
    precipitation = np.stack(precipitation)
    temperature[temperature < -3000] = np.nan
    precipitation[precipitation < -3000] = np.nan

    H, W = temperature.shape[1:]
    j = (np.round((cities['longitude'].to_numpy() + 180) * W / 360) % W).astype(int)  # (n)
    i = (np.minimum(np.floor((-cities['latitude'].to_numpy() + 90) * H / 180), H - 1)).astype(int)  # (n)
    temps = temperature[:, i, j].T  # (n, 12)
    precs = precipitation[:, i, j].T  # (n, 12)

    indices = np.argwhere(~np.isnan(temperature[0]))
    def nearest_non_nan(arr, coords) -> np.ndarray:
        nan = np.any(np.isnan(coords), axis=1)
        i_nan, j_nan = i[nan], j[nan]
        argmin = np.empty((2, i_nan.shape[0]), dtype=int)
        for k, (_i, _j) in tqdm(enumerate(zip(i_nan, j_nan)), total=i_nan.shape[0]):
            argmin[:, k] = indices[np.argmin((indices[:, 0] - _i) ** 2 + (indices[:, 1] - _j) ** 2)]
        return arr[:, argmin[0], argmin[1]]

    temps[np.any(np.isnan(temps), axis=1)] = nearest_non_nan(temperature, temps).T
    precs[np.any(np.isnan(precs), axis=1)] = nearest_non_nan(precipitation, precs).T

    cities[[f'temp{i}' for i in range(12)]] = temps
    cities[[f'prec{i}' for i in range(12)]] = precs


def haversine(coord1: np.ndarray, coord2: np.ndarray) -> np.ndarray:
    # coord1, [2, ...]
    # coord2, [2, ...]
    R = 6371  # Radius of Earth in kilometers
    lat1, lon1 = np.radians(coord1[0]), np.radians(coord1[1])
    lat2, lon2 = np.radians(coord2[0]), np.radians(coord2[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c  # [...]


def min_distances(coords: np.ndarray) -> np.ndarray:
    # coords: [2, n]
    distances = np.empty(coords.shape[1])
    for i in tqdm(range(coords.shape[1])):
        distances[i] = np.min(haversine(coords[:, i], np.delete(coords, i, axis=1)))
    return distances


def fill_temp_prec_distances(cities: pd.DataFrame) -> None:
    selector = [f'temp{i}' for i in range(12)]
    cities['temp_dist'] = min_score_distances(cities[selector].to_numpy().T)
    selector = [f'prec{i}' for i in range(12)]
    cities['prec_dist'] = min_score_distances(cities[selector].to_numpy().T)


def min_score_distances(coords: np.ndarray) -> np.ndarray:
    # coords: [12, n]
    distances = np.empty(coords.shape[1])
    step = 100
    for i in tqdm(range(0, coords.shape[1], step)):
        curr = np.sort(coords[:, i:i + step, None], axis=0)
        other = np.sort(np.delete(coords, np.arange(i, min(i + step, coords.shape[1])), axis=1)[:, None, :], axis=0)
        distances[i:i + step] = np.min(np.mean(np.abs(curr - other), axis=0), axis=1)
    return distances


def filter_csv(cities: pd.DataFrame) -> pd.DataFrame:
    capital = cities[cities['capital'] == 'primary']
    populous_top10 = cities.sort_values(by='population', ascending=False).groupby('country').head(10)
    populous_threshold = cities[cities['population'] > 100000]
    unique = cities[(cities['temp_dist'] > 0.5) | (cities['prec_dist'] > 12) | (cities['distances'] > 75)]
    include = pd.concat([populous_top10, populous_threshold])
    include = include.loc[include.duplicated(keep=False)]
    return pd.concat([include, capital, unique]).drop_duplicates()
