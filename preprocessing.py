"""
Preprocess raw TIFF files into machine learning inputs/targets.
Data Source: https://www.worldclim.org/data/worldclim21.html

Resources:
- https://www.youtube.com/@madelinejameswrites/playlists (https://www.madelinejameswrites.com/blog/air-currents-and-pressure)
- https://www.youtube.com/@Artifexian/videos
- https://docs.google.com/document/d/1E1yWY72XkDQzdzYKUdo0Lg0TmSXRh49cvzvfSEykMuU/edit?tab=t.0#heading=h.g13d333y4wmq

Continentality (0-4 scale):
- 0: latitudes 0-10, extended by enclosed seas, warm currents up to 23.27
    - 100-200 km inland
- 1: latitudes 10-23.27
- 2: latitudes 23.27-90 (only continental land masses)
    - Measure surface area of continents at latitudes 35-70 -> remove 2 if below 4.5M km^2
    - Remove west-coast, first 300-400 km
    - Remove north/south coasts, first 100-250 km
    - Remove areas less than 2000 km wide.
    - If continent doesn't extend past 30, remove it
- 3: latitudes 40-90
    - Measure surface area of continents at latitudes 40-90, remove 3 if below 10M km^2
    - Set to 2 the west-coast, first 1700-2000 km
    - Extend 3 to 35 if vertical distance to ocean is >1000km
    - Extend 3 horizontally at 70-90 latitudes
- 4: latitudes 50-90 on 3
    - Set to 3 the west-coast first 4000 km
    - Take 80-700 km off near seas
- Everything else is 1. Make sure there's a smooth gradient

Pressure zones generally follows 0-30-60 latitudes -> 0/60 is low, 30/90 is high
- At 30-60 degrees, land amplifies the effect of high/low pressure
- Winter -> 30N/60S is extreme than 30S/60N
- Summer -> 30S/60N is extreme than 30N/60S

- Summer -> low pressure at large continents less intense low pressure at polar oceans
- Winter -> low pressure at polar oceans.

Wind directions influenced by the ITCZ 0-30-60 divisions and the pressure/strength
- Map onshore vs offshore winds on coastlines
Ocean currents - map temperature on coastlines

Precipitation (0-1-2-3-4 scale)
- high-pressure + continental + offshore wind regions = 0, everything else = 1
- low-pressure -> +1-2
- onshore winds -> +1. Fetch (land = spread more inland, mountain = less)
- warm sea, onshore winds -> +1
- rainshadow -> +1

If fetch > 100 km, precipitation up to fetch km inland

Elevation vs temperature
- On average -6 C per km
- -5 C (moist area, humid, low-pressure)
- -10 C (dry air, high pressure, elevation > 5000km)

# >>> process_raw_data(1, 'data.npy')
# Saved data to data.npy
# >>> process_raw_data_retrograde('t-retrograde.npy')
# Saved data to t-retrograde.npy
>>> preprocess('data.npy', 'x.npy', 't.npy')
Preprocessing complete at x.npy, t.npy
>>> preprocess_extra(False, True, 'data.npy', 'x')
Preprocessing complete at x-flipped.npy
>>> preprocess_extra(True, False, 'data.npy', 'x')
Preprocessing complete at x-retrograde.npy
>>> preprocess_extra(True, True, 'data.npy', 'x')
Preprocessing complete at x-retrograde-flipped.npy
"""
import numpy as np
# import netCDF4 as nc
# import matplotlib.pyplot as plt
# import os
# from PIL import Image, ImageFile
# from scipy.ndimage import zoom
from typing import Optional, Union
from numpy.fft import fft2, ifft2
from scipy.stats.mstats import gmean
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.ndimage import convolve1d, gaussian_filter, binary_dilation, distance_transform_edt, \
    grey_dilation, binary_fill_holes, binary_closing, binary_erosion, gaussian_filter1d, rotate, label
from scipy.signal import convolve2d
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
# ImageFile.LOAD_TRUNCATED_IMAGES = True
# Image.MAX_IMAGE_PIXELS = None
# LAKE_PATH = r'C:\Users\guoli\Documents\Coding\python stuff\climate\data\elevation'
# ELEVATION_PATH = r'C:\Users\guoli\Documents\Coding\python stuff\climate\data\elevation'
# PRECIPITATION_PATH = r'C:\Users\guoli\Documents\Coding\python stuff\climate\data\precipitation'
# TEMPERATURE_PATH = r'C:\Users\guoli\Documents\Coding\python stuff\climate\data\temperature_average'
# RETROGRADE_PATH = r'C:\Users\guoli\Documents\Coding\python stuff\climate'


####################################################################################################
# Raw Data
####################################################################################################
# def load_image(path: str, resolution: Optional[tuple[int, int]] = None) -> np.ndarray:
#     """Resolution is (width, height)."""
#     image = Image.open(path)
#     if resolution is None:
#         resolution = image.size
#     return np.array(image.resize(resolution, Image.NEAREST))
#
#
# def process_raw_data(pixels_per_degree: int = 1, output_path: str = 'data.npy') -> None:
#     H = round(180 * pixels_per_degree)
#     W = H * 2
#     resolution = (W, H)
#
#     # Elevation, land
#     elevation = load_image(os.path.join(ELEVATION_PATH, 'wc2.1_5m_elev.tif'), resolution).astype(
#         float)
#     land = elevation != -32768
#     lakes = load_image(os.path.join(ELEVATION_PATH, 'lakes.png'), resolution) > 0
#     land[lakes] = False
#     elevation[~land] = np.nan
#
#     # Temperature, precipitation
#     digit = lambda x: f'{"0" * (2 - len(str(x)))}{x}'
#     temp = np.stack(
#         [load_image(os.path.join(TEMPERATURE_PATH, f'wc2.1_5m_tavg_{digit(i)}.tif'), resolution) for
#          i in range(1, 13)])
#     prec = np.stack(
#         [load_image(os.path.join(PRECIPITATION_PATH, f'wc2.1_5m_prec_{digit(i)}.tif'), None) for i
#          in range(1, 13)])
#     prec[prec < -10000] = 0
#     prec = np.maximum(0, np.stack(
#         [zoom(prec[i], H / prec.shape[1], order=3) for i in range(prec.shape[0])]))
#
#     data = np.concatenate([elevation[None], temp, prec], axis=0)  # (25, H, W)
#     np.save(output_path, data)
#     print(f"Saved data to {output_path}")
#
#
# def process_raw_data_retrograde(output_path: str = 't-retrograde.npy') -> None:
#     path = os.path.join(RETROGRADE_PATH, 'ret0001_echam6_BOT_lm_7000-7999.nc')
#     dataset = nc.Dataset(path)
#     temp = dataset.variables['tsurf']
#     prec = dataset.variables['aprc'][:, :] + dataset.variables['aprl'][:, :] + dataset.variables[
#                                                                                    'aprs'][:, :]
#     temp = np.array(temp) - 273.15
#     prec = np.array(prec) * 2628000  # kg/(m^2 s) to mm
#     shift_factor = temp.shape[2] // 2
#     temp = np.roll(temp, shift_factor, axis=2)
#     prec = np.roll(prec, shift_factor, axis=2)
#     data = np.flip(np.concatenate([temp, prec], axis=0), axis=2)
#     np.save(output_path, data)
#     print(f"Saved data to {output_path}")


####################################################################################################
# Gaussian Water Influence
####################################################################################################
def get_longitude_latitude(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    H, W = data.shape
    x, y = np.meshgrid(np.arange(0, 360, 360 / W), np.arange(0, 180, 180 / H))
    longitude = x - 180 + (180 / W)
    latitude = 90 - y - (90 / H)
    return longitude, latitude


def bilinear_interpolation(matrix: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Interpolate matrix[y, x], where y and x are float matrices."""
    assert matrix.shape == x.shape == y.shape, f'{matrix.shape}, {x.shape}, {y.shape}'

    H, W = matrix.shape
    x0 = np.clip(np.floor(x).astype(int), 0, W - 1)
    x1 = np.clip(np.ceil(x).astype(int), 0, W - 1)
    y0 = np.clip(np.floor(y).astype(int), 0, H - 1)
    y1 = np.clip(np.ceil(y).astype(int), 0, H - 1)

    top_left = matrix[y0, x0].astype(float)
    top_right = matrix[y0, x1].astype(float)
    bottom_left = matrix[y1, x0].astype(float)
    bottom_right = matrix[y1, x1].astype(float)

    progress_x = x - x0
    progress_y = y - y0

    left = top_left * (1 - progress_y) + bottom_left * progress_y
    right = top_right * (1 - progress_y) + bottom_right * progress_y
    return left * (1 - progress_x) + right * progress_x


def lcea_projection(map: np.ndarray, inverse: bool = False) -> np.ndarray:
    """Apply the Lambert cylindrical equal-area projection to an
    equirectangular map (or the inverse transformation)."""
    H, W = map.shape
    x, y = np.meshgrid(np.arange(0, 360, 360 / W), np.arange(0, 180, 180 / H))
    longitude = 180 - x - (90 / H)
    latitude = 90 - y - (90 / H)

    if inverse:
        latitude_new = np.sin(latitude * np.pi / 180) * (H / 2)
    else:
        latitude_new = np.arcsin(latitude / 90) * (180 / np.pi)

    x_new = (180 - longitude - (90 / H)) * W / 360
    y_new = (90 - latitude_new - (90 / H)) * H / 180
    return bilinear_interpolation(map, x_new, y_new)


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


def gaussian_blur(matrix: np.ndarray, radius: float, three_dims: bool = False) -> np.ndarray:
    t = round(radius * 2)
    if three_dims:
        matrix_flipped = np.flip(matrix, axis=(1, 2))
        matrix = np.concatenate([matrix_flipped[:, -t:, :], matrix, matrix_flipped[:, :t, :]], axis=1).astype(float)
        matrix = gaussian_filter(matrix, radius, mode=('nearest', 'wrap'), axes=(1, 2))
        return matrix[:, t:-t, :]

    matrix_flipped = np.flip(matrix, axis=(0, 1))
    matrix = np.concatenate([matrix_flipped[-t:, :], matrix, matrix_flipped[:t, :]], axis=0).astype(float)
    matrix = gaussian_filter(matrix, radius, mode=('nearest', 'wrap'), axes=(0, 1))
    return matrix[t:-t, :]


def gaussian_water_influence(land: np.ndarray, locality: Union[float, list[float]],
                             use_lcea: bool = True) -> np.ndarray:
    if isinstance(locality, list):
        return np.stack([gaussian_water_influence(land, x, use_lcea) for x in locality])
    assert locality > 0
    water = 1 - land

    if use_lcea:
        return lcea_projection(gaussian_blur(lcea_projection(water), locality), inverse=True)
    return gaussian_blur(water, locality)


####################################################################################################
# Dilated Water Influence
####################################################################################################
def dilated_edge_distances(land: np.ndarray, locality: Union[int, list[int]] = 7,
                           use_lcea: bool = False) -> np.ndarray:
    """
    Use dilation for more robust edge distances at differing localities/scales.
    """
    if use_lcea:
        land = np.round(lcea_projection(land)).astype(bool)

    max_locality = locality if isinstance(locality, int) else max(locality)
    zeros = np.zeros((1, land.shape[1]), dtype=bool)
    curr_land = np.concatenate([zeros, land, zeros], axis=0)
    coastlines = []
    for _ in range(max_locality):
        coastlines.append(distance_transform_edt(curr_land)[1:-1])
        binary_dilation(curr_land, output=curr_land, border_value=0)

    if use_lcea:
        coastlines = [lcea_projection(coastline, True) for coastline in coastlines]

    if isinstance(locality, list):
        return np.stack([np.mean(coastlines[:i], axis=0) for i in locality])
    return np.mean(coastlines, axis=0)


####################################################################################################
# Elevation Differences
####################################################################################################
def directional_kernel(kernel_radius: int, angle: int, distance: int = -1,
                       angle_strictness: int = 4, distance_tolerance: int = 5) -> np.ndarray:
    """
    Create a convolutional kernel of size (2 * radius + 1, 2 * radius + 1)
    for averaging, with weights centered in a specified angle (0 is right,
    angles move clockwise).

    A distance from the origin can be optionally specified; otherwise, any
    distance will be considered acceptable.
    """
    angle = angle % 360
    assert kernel_radius > 0

    # Calculate the angle of each coordinate in {-radius, ..., radius}^2
    x, y = np.meshgrid(np.arange(-kernel_radius, kernel_radius + 1, 1), np.arange(-kernel_radius, kernel_radius + 1, 1))
    kernel = np.arctan2(x, y) * 180 / np.pi

    # Difference between angles and desired angle
    angle_diff = 180 - abs(180 - (angle - kernel) % 360)
    kernel = ((90 - angle_diff) / 90) ** (2 * angle_strictness + 1)

    # Difference between distance and desired distance
    if distance >= 0:
        distances = np.sqrt(x ** 2 + y ** 2)
        distances = 1 - np.abs((distances - distance) / (distance_tolerance / 2)) ** 0.5
        kernel *= np.maximum(0, distances)

    kernel[kernel_radius, kernel_radius] = 0
    kernel /= np.sum(np.abs(kernel))
    return kernel


def elevation_differences(elevation: np.ndarray, land: np.ndarray, kernel_radius: int, angle: int,
                          distance: int = -1, angle_strictness: int = 4,
                          distance_tolerance: int = 5, method: str = 'fft') -> np.ndarray:
    kernel = directional_kernel(kernel_radius, angle, distance, angle_strictness, distance_tolerance)

    # Single-water pixels are ignored
    more_land = binary_dilation(land, iterations=kernel_radius)
    elevation = interpolate_mask(elevation, land ^ more_land, 'nearest')

    t = kernel_radius * 2
    if method == 'fft':
        elevation = np.concatenate([np.flip(elevation[-t:, :], axis=(0, 1)), elevation,
                                    np.flip(elevation[:t, :], axis=(0, 1))], axis=0)
        elevation = np.concatenate([elevation[:, -t:], elevation, elevation[:, :t]], axis=1)
        map = np.real(ifft2(fft2(elevation) * fft2(kernel, s=elevation.shape)))[t:-t, t:-t]
    elif method == 'convolve':
        elevation = np.concatenate([elevation[:, -t:], elevation, elevation[:, :t]], axis=1)
        map = convolve2d(elevation, kernel, mode='same', boundary='symm')[:, t:-t]
    else:
        raise ValueError(method)
    return map


def gaussian_elevation_differences(elevation: np.ndarray, land: np.ndarray, locality: Union[int, list[int]]) -> np.ndarray:
    if isinstance(locality, list):
        return np.concatenate([gaussian_elevation_differences(elevation, land, i) for i in locality], axis=0)

    # Single-water pixels are ignored
    more_land = binary_dilation(land)
    elevation = interpolate_mask(elevation, land ^ more_land, 'nearest')

    # Horizontal blur
    # horizontal = gaussian_filter1d(elevation, locality, axis=1, order=1, mode='wrap')

    # Pad outer edges
    t = round(locality * 2)
    elevation = np.concatenate([elevation[:, -t:], elevation, elevation[:, :t]], axis=1)
    matrix_flipped = np.flip(elevation, axis=(0, 1))
    elevation = np.concatenate([matrix_flipped[-t:, :], elevation, matrix_flipped[:t, :]], axis=0).astype(float)

    # Diagonal blur, then reset sizzes
    rising_left = rotate(gaussian_filter1d(rotate(elevation, 45), round(locality * 0.7), axis=1, order=1), -45)
    rising_right = rotate(gaussian_filter1d(rotate(elevation, -45), round(locality * 0.7), axis=1, order=1), 45)
    H, W = rising_left.shape
    h, w = land.shape
    rising_left = rising_left[H // 2 - h // 2:H // 2 + h // 2, W // 2 - w // 2:W // 2 + w // 2]
    rising_right = rising_right[H // 2 - h // 2:H // 2 + h // 2, W // 2 - w // 2:W // 2 + w // 2]

    rising_left_mask = ~binary_erosion(more_land, structure=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), iterations=locality + 1)
    rising_right_mask = ~binary_erosion(more_land, structure=np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]), iterations=locality + 1)
    # horizontal[less_land] = 0
    rising_left[rising_left_mask] = 0
    rising_right[rising_right_mask] = 0

    # return np.stack([gaussian_blur(horizontal, locality),
    #                  gaussian_blur(rising_left, locality),
    #                  gaussian_blur(rising_right, locality)])

    return np.stack([-gaussian_blur(rising_left, locality),
                     gaussian_blur(rising_right, locality)])


####################################################################################################
# ITCZ
####################################################################################################
def get_itcz_single_lat(matrix: np.ndarray, lat: np.ndarray, lat_offset: int) -> np.ndarray:
    H = matrix.shape[0]
    land_detail_levels = [H // 60, H // 36, H // 18]
    lat_prox_tolerance = 30
    lat_dist_tolerance = 25
    peak_blur = H // 18
    land_percent_threshold = 0.025

    window_function = lambda x, a, b: np.maximum(0, -4 * (x - a) * (x - b) / (a - b) ** 2)

    land_weight = np.mean([gaussian_blur(matrix, x) for x in land_detail_levels], axis=0)
    proximity_weight = window_function(lat, lat_offset - lat_prox_tolerance, lat_offset + lat_prox_tolerance)
    weights = land_weight * proximity_weight
    gradient = weights - np.maximum(np.roll(weights, 1, axis=0), np.roll(weights, -1, axis=0))
    peaks = proximity_weight * ((gradient > 0) & matrix & (np.abs(lat - lat_offset) < 20))
    peaks = gaussian_blur(peaks, peak_blur)

    itcz = np.argmax(peaks, axis=0).astype(float)
    land_influence = np.sum(peaks, axis=0)
    itcz[land_influence < land_percent_threshold] = 90 - lat_offset
    itcz = np.clip(itcz, 90 - (lat_offset + lat_dist_tolerance), 90 - (lat_offset - lat_dist_tolerance))
    itcz = convolve1d(itcz, np.ones(H // 6) / (H // 6), mode='wrap')
    itcz = convolve1d(itcz, np.ones(H // 18) / (H // 18), mode='wrap')
    return itcz  # (W)


def get_itcz(land: np.ndarray, lat: np.ndarray, lat_offset: list[float]) -> np.ndarray:
    H = land.shape[0]
    land_detail_levels = [H // 60, H // 36, H // 18]
    lat_prox_tolerance = 30
    lat_dist_tolerance = 25
    peak_blur = H // 18
    land_percent_threshold = 0.025

    lat_offset = np.array(lat_offset)[:, None, None]
    window_function = lambda x, a, b: np.maximum(0, -4 * (x - a) * (x - b) / (a - b) ** 2)

    # Get peaks
    land_weight = np.mean([gaussian_blur(land, x) for x in land_detail_levels], axis=0)[None]
    proximity_weight = window_function(lat[None], lat_offset - lat_prox_tolerance, lat_offset + lat_prox_tolerance)
    weights = land_weight * proximity_weight
    gradient = weights - np.maximum(np.roll(weights, 1, axis=1), np.roll(weights, -1, axis=1))
    peaks = proximity_weight * ((gradient > 0) & land & (np.abs(lat - lat_offset) < 20))
    peaks = gaussian_blur(peaks, peak_blur, three_dims=True)

    # Take center of peaks, apply smoothing
    itcz = np.argmax(peaks, axis=1).astype(float)
    land_influence = np.sum(peaks, axis=1)
    lat_offset = lat_offset[:, 0]
    for i in range(len(lat_offset)):
        itcz[i][land_influence[i] < land_percent_threshold] = 90 - lat_offset[i]
    itcz = np.clip(itcz, 90 - (lat_offset + lat_dist_tolerance), 90 - (lat_offset - lat_dist_tolerance))
    itcz = convolve1d(itcz, np.ones(H // 6) / (H // 6), mode='wrap', axis=1)
    itcz = convolve1d(itcz, np.ones(H // 18) / (H // 18), mode='wrap', axis=1)
    return itcz  # (k, W) where len(lat_offset) == k and land.shape[1] == W


def interpolate_mask(data: np.ndarray, mask: Union[str, np.ndarray], interpolator: str) -> np.ndarray:
    """Interpolate the nan values of a 2D numpy array."""
    if isinstance(mask, str) and mask == 'nan':
        point_mask = np.isnan(data)  #  points the interpolator doesn't look at
        mask = point_mask  # points to fill
    else:
        point_mask = mask | np.isnan(data)

    points = np.stack(np.where(~point_mask)).T
    values = data[points[:, 0], points[:, 1]]
    if interpolator == 'linear':
        interpolator = LinearNDInterpolator(points, values)
    elif interpolator == 'nearest':
        interpolator = NearestNDInterpolator(points, values)
    else:
        raise ValueError()

    i = np.stack(np.where(mask)).T
    interpolated = interpolator(i)
    data = data.copy()
    data[mask] = interpolated
    return data


def get_itcz_map(land: np.ndarray, lat: np.ndarray) -> np.ndarray:
    H, W = land.shape
    offsets = [-40] + list(range(-30, 31, 5)) + [40]
    itcz = np.round(get_itcz(land, lat, offsets)).astype(int)

    map = np.full((len(offsets), H, W), np.nan, dtype=float)
    x = np.arange(W)
    for i in range(len(offsets)):
        map[i, itcz[i], x] = offsets[i]
    map = np.nanmean(map, axis=0)
    map[0] = 90
    map[-1] = -90
    map = interpolate_mask(map, 'nan', 'linear')
    return gaussian_blur(map, H // 90)


####################################################################################################
# Extras
####################################################################################################
def gradients(land: np.ndarray, blur: Union[int, list[int]]) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(blur, list):
        dys, dxs = [], []
        for x in blur:
            dy, dx = gradients(land, x)
            dys.append(dy)
            dxs.append(dx)
        return np.stack(dys), np.stack(dxs)
    dy, dx = np.gradient(gaussian_blur(land, blur))
    return dy, dx


def get_wind_onshore_offshore(wind_dy: np.ndarray, wind_dx: np.ndarray,
                              dy: np.ndarray, dx: np.ndarray, elevation: np.ndarray) -> np.ndarray:
    direction = wind_dy * dy + wind_dx * dx

    # high-elevation should block air currents
    e = np.log(elevation + 1)
    e[np.isnan(e)] = 0
    radius = 30
    for dir in [0, 90, 180, 270]:
        # for each pixel, find maximum elevation in a given direction
        # e[i, j] / max elevation[i, j] determines the reduction in water influence
        kernel = np.maximum(0, directional_kernel(radius, dir))
        kernel = kernel > np.mean(kernel)
        factor = e / grey_dilation(e, footprint=kernel)
        factor[(e == 0) | np.isinf(factor)] = 0
        np.clip(factor, 0, 1, out=factor)
        direction *= factor
    return direction * 100


def get_wind_directions(itcz: np.ndarray, latitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    H, W = latitude.shape
    latitude_itcz = itcz + latitude

    wind_dy = np.zeros((H, W))
    wind_dx = np.zeros((H, W))
    for i, boundary in enumerate([-60, -30, 30, 60]):
        map = gaussian_blur(latitude_itcz <= boundary, W / 50)
        if i % 2 == 1:
            map *= -1
        wind_dx -= map
    for i, boundary in enumerate([-60, -30, 0, 30, 60]):
        map = gaussian_blur(latitude_itcz <= boundary, W / 50)
        if i % 2 == 1:
            map *= -1
        wind_dy -= map
    return wind_dy * 2 + 1, wind_dx * 2 - 1


def get_water_current_temperature(dx: np.ndarray, elevation: np.ndarray,
                                  latitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # 60/90 (clockwise), moving left = warm
    # 30/60 (counter-clockwise), moving right = warm
    # 0/30 (clockwise), moving left = warm
    # -30/0 (counter-clockwise), moving left = warm
    # -60/-30 (clockwise), moving right = warm
    # -90/-60 (counter-clockwise), moving left = warm
    W = latitude.shape[1]
    boundary = (latitude > 60) | (latitude < -60) | ((latitude < 30) & (latitude > 2.5)) | (
                (latitude < -2.5) & (latitude > -30))
    boundary = 2 * gaussian_blur(boundary, W / 50) - 1
    boundary2 = ((latitude < 30) & (latitude > 2.5)) | ((latitude < -2.5) & (latitude > -30))
    boundary2 = 2 * gaussian_blur(boundary2, W / 50) - 1

    temperature = -boundary * dx  # (positive = warm, negative = cold)
    temperature2 = -boundary2 * dx

    # high-elevation should block water currents
    e = np.sqrt(elevation)
    e[np.isnan(e)] = 0
    radius = 30
    for dir in [90, 270]:
        # for each pixel, find maximum elevation in a given direction
        # e[i, j] / max elevation[i, j] determines the reduction in water influence
        kernel = np.maximum(0, directional_kernel(radius, dir))
        kernel = kernel > np.mean(kernel)
        factor = e / grey_dilation(e, footprint=kernel)
        factor[(e == 0) | np.isinf(factor)] = 0
        np.clip(factor, 0, 1, out=factor)
        temperature *= factor
        temperature2 *= factor

    closest_coast = gaussian_blur(np.cbrt(temperature2), W // 100)
    return temperature * 100, closest_coast * 10


def detect_polar_front(land: np.ndarray, latitude: np.ndarray, dx: np.ndarray) -> np.ndarray:
    # Smoothing
    w = land.shape[1] // 2
    l = binary_closing(np.concatenate([land[:, -w:], land, land[:, :w]], axis=1), border_value=1)[:, w:-w]

    # Removal of single points
    l &= np.roll(l, 1, axis=1) | np.roll(l, -1, axis=1) \
         | np.concatenate([l[1:], l[:1]], axis=0) | np.concatenate([l[-1:], l[:-1]], axis=0)

    # Polar, large areas far from west coast
    l &= np.abs(latitude) > 35
    west = binary_dilation(land, border_value=0) & (dx[3] > 0.002)
    west_area = count_region_area(west)
    l &= ~(west_area > 300)
    l_total = l[l].shape[0]
    area = count_region_area(l) > 0.1 * l_total
    dli = dilated_edge_distances(binary_dilation(land, border_value=0, iterations=2), 7, use_lcea=False)
    polar = gaussian_blur(area, 5) * dli
    return polar


def detect_coast(elevation: np.ndarray, land: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dx = np.zeros_like(land, dtype=float)
    w = land.shape[1] // 2
    l = np.concatenate([land[:, -w:], land, land[:, :w]], axis=1)
    for i in range(1, 5):
        base = binary_closing(l, iterations=i, border_value=0)
        dx += gaussian_filter1d(base.astype(float), 15, order=1, axis=1)[:, w:-w]
    dx /= 4

    e = np.log(elevation + 1)
    e[np.isnan(e)] = 0
    radius = 30
    for dir in [90, 270]:
        # for each pixel, find maximum elevation in a given direction
        # e[i, j] / max elevation[i, j] determines the reduction in water influence
        kernel = np.maximum(0, directional_kernel(radius, dir))
        kernel = kernel > np.mean(kernel)
        factor = e / grey_dilation(e, footprint=kernel)
        factor[(e == 0) | np.isinf(factor)] = 0
        np.clip(factor, 0, 1, out=factor)
        dx *= factor
    west = 100 * np.mean([gaussian_blur(np.maximum(dx, 0), i) for i in [3, 5]], axis=0)
    east = 100 * np.mean([gaussian_blur(-np.minimum(dx, 0), i) for i in [3, 5]], axis=0)
    return west, east


####################################################################################################
# Continentality
####################################################################################################
def count_width(arr: np.ndarray) -> np.ndarray:
    H, W = arr.shape
    width = np.zeros_like(arr, dtype=int)
    differences = np.concatenate([np.diff(arr, axis=1), arr[:, :1] ^ arr[:, -1:]], axis=1)
    for i in range(H):
        streaks = np.where(differences[i])[0]
        if np.all(arr[i]):
            width[i] = W
            continue
        if len(streaks) == 0:
            continue
        offset = np.where(arr[i][streaks] == 0)[0][0]
        for j in range(0, len(streaks), 2):
            start = (streaks[(j + offset) % len(streaks)] + 1) % W
            end = (streaks[(j + offset + 1) % len(streaks)] + 1) % W
            if end <= start:
                width[i, start:] = end - start + W
                width[i, :end] = end - start + W
            else:
                width[i, start:end] = end - start
    return width


def count_left_dist(arr: np.ndarray) -> np.ndarray:
    H, W = arr.shape
    left_dist = np.zeros_like(arr, dtype=int)
    differences = np.concatenate([np.diff(arr, axis=1), arr[:, :1] ^ arr[:, -1:]], axis=1)
    for i in range(H):
        streaks = np.where(differences[i])[0]
        if np.all(arr[i]):
            left_dist[i] = W
            continue
        if len(streaks) == 0:
            continue
        offset = np.where(arr[i, streaks] == 0)[0][0]
        for j in range(0, len(streaks), 2):
            start = (streaks[(j + offset) % len(streaks)] + 1) % W
            end = (streaks[(j + offset + 1) % len(streaks)] + 1) % W
            if end <= start:
                k = W - start
                left_range = np.arange(1, W - start + end + 1, 1)
                left_dist[i, start:] = left_range[:k]
                left_dist[i, :end] = left_range[k:]
            else:
                left_dist[i, start:end] = np.arange(1, end - start + 1, 1)
    return left_dist


def count_region_area(land: np.ndarray, structure: Optional[np.ndarray] = None) -> np.ndarray:
    w = land.shape[1] // 2
    regions = np.concatenate([land[:, -w:], land, land[:, :w]], axis=1)
    arr, regions = label(regions, structure=structure)
    for i in range(1, regions + 1):
        arr[arr == i] = arr[arr == i].shape[0]
    return arr[:, w:-w]


def get_islands(land: np.ndarray) -> np.ndarray:
    islands = np.minimum(500, count_region_area(land))
    islands_diag = np.minimum(500, count_region_area(land, np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])))
    return 500 - gmean([islands, islands_diag], axis=0)


def get_continentality(latitude: np.ndarray, x_distortion: np.ndarray, land: np.ndarray,
                       water_current_temperature: np.ndarray, dli: np.ndarray, dwi: np.ndarray,
                       islands: np.ndarray) -> np.ndarray:
    H, W = latitude.shape
    c = np.zeros((5, H, W))
    abs_latitude = np.abs(latitude)

    # TODO use scipy.ndimage.label to count areas

    # hyperoceanic
    region = np.concatenate([land[:, -W // 2:], land, land[:, :W // 2]], axis=1)
    inland_sea = (~region & binary_fill_holes(region)).astype(float)
    land_generous = region.copy()
    for i in range(1, 5):
        binary_dilation(land_generous, output=land_generous, border_value=0)
        inland_sea += binary_dilation(~region & ~land_generous & binary_fill_holes(land_generous),
                                      iterations=i, border_value=0) / i
    inland_sea = gaussian_blur(inland_sea[:, W // 2:-W // 2], 4)

    c[0] += gaussian_blur(abs_latitude <= 10, 5)  # -10/10 latitude
    extension = land & (abs_latitude <= 23.5)
    c[0] += gaussian_blur((dli <= 5) & (water_current_temperature > 0) & extension, 5)  # warm water
    c[0] += gaussian_blur((dwi[0] > 2) & extension, 2)  # small islands
    dli_fine = dilated_edge_distances(land, 2, use_lcea=True) * x_distortion
    c[0] += gaussian_blur(inland_sea * ((dli_fine <= 2) & extension), 2)  # inland seas

    # oceanic
    c[1] += gaussian_blur(((10 < abs_latitude) & (abs_latitude <= 23.5)), 5)
    c[1] += islands / 1000
    # c[1] += gaussian_blur((abs_latitude > 23.5) & ~land, 5) * 0.5

    # More kind towards small water bodies, extremer gradients
    inland_sea_2 = dilated_edge_distances(land, 7) * x_distortion / 7
    inland_sea_2 = gaussian_blur(inland_sea_2 * ~land, 3)

    # subcontinental
    keep = (23.5 < abs_latitude) & land
    threshold_distance = 2000 / (x_distortion * 111)
    width = count_width(binary_dilation(keep, iterations=2, border_value=0))
    keep = width >= threshold_distance  # >= 2000 km width
    threshold_distance = 350 / (x_distortion * 111)
    left_dist = count_left_dist(keep)
    keep = left_dist >= threshold_distance  # >= 350 km from west coast
    c[2] += gaussian_blur(keep, 5)
    c[2] = np.minimum(1, c[2] + gaussian_blur((abs_latitude < 30) & (abs_latitude >= 15) & (dli > 5), 3))  # Extend to 15 latitude if inland
    c[2] -= inland_sea * 0.5 * gaussian_blur((abs_latitude < 70), 3)  # subtract from watery places
    c[2, :, :] = np.maximum(c[2], 0)

    # continental
    keep = (abs_latitude > 40) & land
    left_dist = count_left_dist(binary_dilation(keep, iterations=2, border_value=0))
    threshold_distance = 1700 / (x_distortion * 111)
    condition = (left_dist >= threshold_distance) | ((abs_latitude > 70) & (dli > 1))
    c[3] += gaussian_blur(condition, 5)  # >= 2000 km from west coast
    c[3] += gaussian_blur((abs_latitude >= 35) & (dli > 10), 5)  # >= 35 latitude if >= 1000 km from water
    c[3] -= inland_sea_2 * gaussian_blur((abs_latitude < 70), 3)
    c[3, :, :] = np.maximum(c[3], 0)

    # hypercontinental
    keep = (abs_latitude > 50) & land & (dli_fine > 1)
    left_dist = count_left_dist(binary_dilation(keep, iterations=2, border_value=0))
    threshold_distance = 4000 / (x_distortion * 111)
    c[4] += gaussian_blur(left_dist >= threshold_distance, 5)  # >= 4000 km from west coast
    c[4] -= inland_sea
    c[4] = np.maximum(0, c[4])

    confidence = np.minimum(1, np.sum(c, axis=0))
    c[1] += (x_distortion * (1 - confidence)) * gaussian_blur(abs_latitude < 60, 5)

    # Most likely climate zone
    c[3] = gaussian_blur(c[3] > 0.1, 5)
    c[2] -= c[3]
    c[3] -= c[4]
    # prediction = np.argmax(c, axis=0)
    # return gaussian_blur(prediction, 4)
    return c


####################################################################################################
# Preprocess
####################################################################################################
def concatenate(arrays: list[np.ndarray]) -> np.ndarray:
    data = []
    for arr in arrays:
        if len(arr.shape) == 2:
            data.append(arr[None])
        else:
            data.append(arr)
    return np.concatenate(data, axis=0)


def preprocess(path: str = 'data.npy', x_path: str = 'x.npy', t_path: str = 't.npy') -> None:
    data = np.load(path)
    elevation, temp, prec = data[0], data[1:13], data[13:25]
    _, latitude = get_longitude_latitude(elevation)
    land = ~np.isnan(elevation)
    elevation[~land] = 0
    x_distortion = np.cos(latitude * np.pi / 180)

    # TO DO add in future -> predictions of pressure, wind x/y, temperature

    gwi = gaussian_water_influence(land, [3, 5, 7, 9, 11], use_lcea=True)
    gwi[1:] = np.diff(gwi, axis=0)
    dwi = dilated_edge_distances(~land, [3, 5, 7]) * x_distortion
    dwi[0] -= 1
    dwi[1] -= 1.5
    dwi[2] -= 2
    dwi[:] = np.maximum(0, dwi)
    dwi[1:] = np.diff(dwi, axis=0)
    dli = dilated_edge_distances(land, 3, use_lcea=True) * x_distortion
    itcz = get_itcz_map(land, latitude)
    itcz = itcz - latitude
    ediff_params = [(3, 45, 1), (3, 315, 1), (5, 45, 3), (5, 315, 3)]
    ediff = np.stack([elevation_differences(elevation, land, r, a, d) for r, a, d in ediff_params])
    ediff *= x_distortion
    dy, dx = gradients(land, [1, 3, 5, 7])
    dy *= x_distortion
    dx *= x_distortion
    wind_dy, wind_dx = get_wind_directions(itcz, latitude)
    wind_onshore_offshore = get_wind_onshore_offshore(wind_dy, wind_dx, dy[3], dx[3], elevation)
    water_current_temperature, closest_coast = get_water_current_temperature(dx[3], elevation, latitude)
    closest_coast *= x_distortion
    continentality = get_continentality(latitude, x_distortion, land, water_current_temperature,
                                        dli, dwi, get_islands(land))
    west, east = detect_coast(elevation, land)

    inputs = concatenate([elevation, latitude, land, gwi, dwi, dli, itcz, wind_onshore_offshore,
                          water_current_temperature, closest_coast, continentality, ediff, west, east])
    targets = concatenate([temp, prec])
    np.save(x_path, inputs)
    np.save(t_path, targets)
    print(f"Preprocessing complete at {x_path}, {t_path}")


def preprocess_inference(elevation: np.ndarray, land: np.ndarray) -> np.ndarray:
    elevation[~land] = 0
    _, latitude = get_longitude_latitude(elevation)
    x_distortion = np.cos(latitude * np.pi / 180)

    gwi = gaussian_water_influence(land, [3, 5, 7, 9, 11], use_lcea=True)
    gwi[1:] = np.diff(gwi, axis=0)
    dwi = dilated_edge_distances(~land, [3, 5, 7]) * x_distortion
    dwi[0] -= 1
    dwi[1] -= 1.5
    dwi[2] -= 2
    dwi[:] = np.maximum(0, dwi)
    dwi[1:] = np.diff(dwi, axis=0)
    dli = dilated_edge_distances(land, 3, use_lcea=True) * x_distortion
    itcz = get_itcz_map(land, latitude)
    itcz = itcz - latitude
    ediff_params = [(3, 45, 1), (3, 315, 1), (5, 45, 3), (5, 315, 3)]
    ediff = np.stack([elevation_differences(elevation, land, r, a, d) for r, a, d in ediff_params])
    ediff *= x_distortion
    dy, dx = gradients(land, [1, 3, 5, 7])
    dy *= x_distortion
    dx *= x_distortion
    wind_dy, wind_dx = get_wind_directions(itcz, latitude)
    wind_onshore_offshore = get_wind_onshore_offshore(wind_dy, wind_dx, dy[3], dx[3], elevation)
    water_current_temperature, closest_coast = get_water_current_temperature(dx[3], elevation, latitude)
    closest_coast *= x_distortion
    continentality = get_continentality(latitude, x_distortion, land, water_current_temperature,
                                        dli, dwi, get_islands(land))
    west, east = detect_coast(elevation, land)

    return concatenate([elevation, latitude, land, gwi, dwi, dli, itcz, wind_onshore_offshore,
                        water_current_temperature, closest_coast, continentality, ediff, west, east])


def preprocess_extra(retrograde: bool, flip: bool, path: str = 'data.npy', x_prefix: str = 'x') -> None:
    data = np.load(path)
    elevation = data[0]
    suffix = ''
    if retrograde:
        elevation = np.flip(elevation, axis=1)
        suffix += '-retrograde'
    if flip:
        elevation = np.flip(elevation, axis=(0, 1))
        suffix += '-flipped'
    land = ~np.isnan(elevation)
    data = preprocess_inference(elevation, land)
    x_path = f'{x_prefix}{suffix}.npy'
    np.save(x_path, data)
    print(f"Preprocessing complete at {x_path}")


####################################################################################################
# Postprocess
####################################################################################################
def get_koppen(temp: np.ndarray, prec: np.ndarray, land: np.ndarray) -> np.ndarray:
    h = land.shape[0] // 2
    summer = lambda x: np.concatenate([x[3:9, :h, :], x[[0, 1, 2, 9, 10, 11], h:, :]], axis=1)
    winter = lambda x: np.concatenate([x[[0, 1, 2, 9, 10, 11], :h, :], x[3:9, h:, :]], axis=1)

    apply = lambda x, funcs: [f(x, axis=0) for f in funcs]
    temp_mean, temp_max, temp_min = apply(temp, [np.mean, np.max, np.min])
    prec_sum, prec_min = apply(prec, [np.sum, np.min])
    prec_summer, prec_winter = summer(prec), winter(prec)
    prec_winter_max, prec_winter_min = apply(prec_winter, [np.max, np.min])
    prec_summer_sum, prec_summer_max, prec_summer_min = apply(prec_summer, [np.sum, np.max, np.min])

    B_prec_percent = (prec_summer_sum + 0.0001) / (prec_sum + 0.0001)
    B_threshold = (20 * temp_mean + 280 * (B_prec_percent >= 0.7) + 140 * (B_prec_percent < 0.7) * (B_prec_percent >= 0.3))

    A_threshold = 100 - (prec_sum / 25)
    A = (temp_min >= 18) & (prec_sum > B_threshold)
    Af = A & (prec_min >= 60)
    Am = A & (prec_min < 60) & (prec_min >= A_threshold)
    As = A & (prec_summer_min < A_threshold)
    Aw = A & (prec_winter_min < A_threshold)
    del A, A_threshold

    B = (temp_max >= 10) & (prec_sum <= B_threshold)
    BS = B & (prec_sum >= 0.5 * B_threshold)
    BW = B & (prec_sum < 0.5 * B_threshold)
    B_h = temp_mean >= 18
    BSh, BSk = BS & B_h, BS & ~B_h
    BWh, BWk = BW & B_h, BW & ~B_h
    del B_prec_percent, B, B_h, BS, BW

    _a = temp_max >= 22
    _b = ~_a & (np.sum(temp >= 10, axis=0) >= 4)
    _c = ~_a & ~_b & (temp_min >= -38)
    _d = ~_a & ~_b & ~_c
    _w = (prec_summer_max >= 10 * prec_winter_min)
    _s = (prec_winter_max >= 3 * prec_summer_min) * (prec_summer_min < 40)
    _f = ~_w & ~_s

    C = (temp_min >= 0) & (temp_min < 18) & (temp_max >= 10) & (prec_sum > B_threshold)
    Cw, Cs, Cf = C & _w, C & _s, C & _f
    Cfa, Cfb, Cfc = Cf & _a, Cf & _b, Cf & _c
    Csa, Csb, Csc = Cs & _a, Cs & _b, Cs & _c
    Cwa, Cwb, Cwc = Cw & _a, Cw & _b, Cw & _c
    del C, Cw, Cs, Cf

    D = (temp_min < 0) & (temp_max >= 10) & (prec_sum > B_threshold)
    Dw, Ds, Df = D & _w, D & _s, D & _f
    Dfa, Dfb, Dfc, Dfd = Df & _a, Df & _b, Df & _c, Df & _d
    Dsa, Dsb, Dsc, Dsd = Ds & _a, Ds & _b, Ds & _c, Ds & _d
    Dwa, Dwb, Dwc, Dwd = Dw & _a, Dw & _b, Dw & _c, Dw & _d
    del D, Dw, Ds, Df, _w, _s, _f, _a, _b, _c, _d, B_threshold

    ET = (temp_max < 10) & (temp_max >= 0)
    EF = temp_max < 0

    c = np.full(land.shape, -1.)
    c[Af], c[Am], c[As], c[Aw] = range(0, 4)
    c[BSh], c[BSk], c[BWh], c[BWk] = range(4, 8)
    c[Cfa], c[Cfb], c[Cfc], c[Csa], c[Csb], c[Csc], c[Cwa], c[Cwb], c[Cwc] = range(8, 17)
    c[Dfa], c[Dfb], c[Dfc], c[Dfd] = range(17, 21)
    c[Dsa], c[Dsb], c[Dsc], c[Dsd] = range(21, 25)
    c[Dwa], c[Dwb], c[Dwc], c[Dwd] = range(25, 29)
    c[EF], c[ET] = range(29, 31)
    return c


def get_trewartha(temp: np.ndarray, prec: np.ndarray, elevation: np.ndarray, latitude: np.ndarray,
                  land: np.ndarray) -> np.ndarray:
    h = land.shape[0] // 2
    summer = lambda x: np.concatenate([x[3:9, :h, :], x[[0, 1, 2, 9, 10, 11], h:, :]], axis=1)
    winter = lambda x: np.concatenate([x[[0, 1, 2, 9, 10, 11], :h, :], x[3:9, h:, :]], axis=1)
    apply = lambda x, funcs: [f(x, axis=0) for f in funcs]
    temp_mean, temp_max, temp_min = apply(temp, [np.mean, np.max, np.min])
    prec_sum, prec_min = apply(prec, [np.sum, np.min])
    prec_summer, prec_winter = summer(prec), winter(prec)
    prec_winter_min = np.min(prec_winter, axis=0)
    prec_summer_sum, prec_summer_max = apply(prec_summer, [np.sum, np.max])

    temp_gt_10 = np.sum(temp >= 10, axis=0)
    prec_threshold = 10 * (temp_mean - 10) + 300 * (prec_summer_sum + 0.0001) / (prec_sum + 0.0001)

    A = (temp_min >= 18) & (prec_sum >= 2 * prec_threshold)
    Ar = A & (np.sum(prec >= 60, axis=0) > 10)
    Am = A & ~Ar & (prec_min < 60) & (prec_min >= (2500 - prec_sum) / 25)
    Aw = A & ~Ar & ~Am & (np.sum(winter(prec) < 60, axis=0) > 2)
    As = A & ~Ar & ~Aw & ~Aw

    B = (prec_sum < 2 * prec_threshold) & (temp_gt_10 >= 3)
    BS = B & (prec_sum >= prec_threshold)
    BSh = BS & (temp_gt_10 >= 8)
    BSk = BS & ~BSh
    BW = B & ~BS
    BWh = BW & (temp_gt_10 >= 8)
    BWk = BW & ~BWh
    del BS, BW

    _a = temp_max >= 22
    _b = ~_a
    _w = (prec_sum < 890) & (prec_winter_min < 30) & (prec_winter_min < prec_summer_max / 3)
    _s = (prec_sum < 890) & (prec_winter_min < 30) & (prec_winter_min < prec_summer_max / 3)
    _f = ~_s & ~_w

    C = (temp_gt_10 >= 8) & (prec_sum >= 2 * prec_threshold) & (temp_min < 18)
    Cf, Cw, Cs = C & _f, C & _w, C & _s
    Cfa, Cfb = Cf & _a, Cf & _b
    Cwa, Cwb = Cw & _a, Cw & _b
    Csa, Csb = Cs & _a, Cs & _b
    del Cf, Cw, Cs

    D = (temp_gt_10 < 8) & (temp_gt_10 >= 4) & (prec_sum >= 2 * prec_threshold)
    DC = D & (temp_min < 0)
    DO = D & ~DC
    DCfa, DCfb = DC & _f & _a, DC & _f & _b
    DCsa, DCsb = DC & _s & _a, DC & _s & _b
    DCwa, DCwb = DC & _w & _a, DC & _w & _b
    DOfa, DOfb = DO & _f & _a, DO & _f & _b
    DOsa, DOsb = DO & _s & _a, DO & _s & _b
    DOwa, DOwb = DO & _w & _a, DO & _w & _b
    del DC, DO

    E = (temp_gt_10 < 4) & (temp_gt_10 >= 1)
    EC = E & (temp_min < -10)
    EO = E & (temp_min >= -10)
    F = temp_max < 10
    Ft = F & (temp_max > 0)
    Fi = F & (temp_max < 0)
    del _a, _b, _w, _s, _f

    new_temp = temp + 0.0056 * elevation * gaussian_blur(np.abs(latitude) < 65, 5)
    new_temp_max = np.max(new_temp, axis=0)
    new_temp_min = np.min(new_temp, axis=0)
    new_temp_gt_10 = np.sum(new_temp >= 10, axis=0)
    new_A = (new_temp_min >= 18) * (prec_sum >= 2 * prec_threshold)
    new_B = (prec_sum < 2 * prec_threshold) * (new_temp_gt_10 >= 3)
    new_C = (new_temp_gt_10 >= 8) * (prec_sum >= 2 * prec_threshold) * (new_temp_min < 18)
    new_D = (new_temp_gt_10 < 8) * (new_temp_gt_10 >= 4) * (prec_sum >= 2 * prec_threshold)
    new_E = (new_temp_gt_10 < 4) * (new_temp_gt_10 >= 1)
    new_F = new_temp_max < 10
    del new_temp, new_temp_max, new_temp_min, new_temp_gt_10

    c = np.full(land.shape, -1.)
    c[Ar], c[Am], c[Aw], c[As] = range(4)
    c[BSh], c[BSk], c[BWh], c[BWk] = range(4, 8)
    c[Cfa], c[Cfb], c[Cwa], c[Cwb], c[Csa], c[Csb] = range(8, 14)
    c[DCfa], c[DCfb], c[DCsa], c[DCsb], c[DCwa], c[DCwb] = range(14, 20)
    c[DOfa], c[DOfb], c[DOsa], c[DOsb], c[DOwa], c[DOwb] = range(20, 26)
    c[EC], c[EO] = 26, 27
    c[Ft], c[Fi] = 28, 29
    different_zone = np.logical_or.reduce([A ^ new_A, B ^ new_B, C ^ new_C, D ^ new_D, E ^ new_E, F ^ new_F])
    c[different_zone & (elevation >= 2500)] = 30
    del A, B, C, D, E, F, new_A, new_B, new_C, new_D, new_E, new_F
    return c


def get_elevation_extremes(elevation: np.ndarray, land: np.ndarray) -> tuple[list[float], list[float]]:
    if np.all(~land):
        return [-1, -1, -1], [-1, -1, -1]

    e = elevation.copy()
    e[~land] = -np.inf
    i, j = divmod(np.argmax(e), e.shape[1])
    max_elevation = [int(i), int(j), float(e[i, j])]

    e[~land] = np.inf
    i, j = divmod(np.argmin(e), e.shape[1])
    min_elevation = [int(i), int(j), float(e[i, j])]
    return min_elevation, max_elevation


def get_distance_extremes(land: np.ndarray, latitude: np.ndarray) -> tuple[list[float], list[float]]:
    if np.all(land) or np.all(~land):
        return [-1, -1, -1], [-1, -1, -1]
    x_distortion = np.repeat(np.cos(latitude * np.pi / 180), 2, axis=1)

    W = land.shape[1]
    w = W // 2
    region = np.concatenate([land[:, -w:], land, land[:, :w]], axis=1)

    land_distance = lcea_projection(distance_transform_edt(lcea_projection(region)), True) * x_distortion
    i, j = divmod(np.argmax(land_distance), land_distance.shape[1])
    out1 = [float(i), float((j + w) % W), float(land_distance[i, j] * 111)]

    water_distance = lcea_projection(distance_transform_edt(lcea_projection(~region)), True) * x_distortion
    i, j = divmod(np.argmax(water_distance), water_distance.shape[1])
    out2 = [float(i), float((j + w) % W), float(water_distance[i, j] * 111)]
    return out1, out2
