"""Climate Net - GUI Part"""

from flask import Flask, render_template, request, jsonify
from preprocessing import preprocess_inference, get_elevation_extremes, get_distance_extremes, \
    get_koppen, get_trewartha
from model_prec import PrecipitationNet
from model_temp import TemperatureNet
from datetime import datetime
import numpy as np
import webview
import traceback

TEMPERATURE_NET = TemperatureNet()
PRECIPITATION_NET = PrecipitationNet()
APP = Flask(__name__)


@APP.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    t0 = datetime.now()
    print(f"[1/4] Receiving data...")
    data = request.get_json()
    elevation = np.array(data['elevation'], dtype=float)
    land = ~np.array(data['water'], dtype=bool)

    # Preprocess
    t = datetime.now() - t0
    print(f"[2/4] Preprocessing data... ({t.seconds + t.microseconds / 1e6} seconds elapsed)")
    x = preprocess_inference(elevation, land)
    latitude = x[1]

    # Predict
    t = datetime.now() - t0
    print(f"[3/4] Making predictions... ({t.seconds + t.microseconds / 1e6} seconds elapsed)")
    temp = TEMPERATURE_NET.predict(x)[:12]
    prec = PRECIPITATION_NET.predict(x, temp)[:12]

    # Postprocessing
    temp_min, temp_max = float(np.min(temp)), float(np.max(temp))
    prec_min, prec_max = float(np.min(prec)), float(np.max(prec))
    koppen = get_koppen(temp, prec, land)
    trewartha = get_trewartha(temp, prec, elevation, latitude, land)

    # Statistics
    min_elevation, max_elevation = get_elevation_extremes(elevation, land)
    farthest_land, farthest_water = get_distance_extremes(land, latitude)

    t = datetime.now() - t0
    print(f"[4/4] Sending predictions... ({t.seconds + t.microseconds / 1e6} seconds elapsed)")
    return jsonify(temp=temp.tolist(), prec=prec.tolist(), koppen=koppen.tolist(), water=data['water'],
                   trewartha=trewartha.tolist(), temp_range=[temp_min, temp_max], prec_range=[prec_min, prec_max],
                   statistics=[min_elevation, max_elevation, farthest_land, farthest_water])


webview.create_window('Climate Net', APP, min_size=(900, 500))

# GUI mode - close splash screen on load
try:
    import pyi_splash
    pyi_splash.close()
except:
    pass

if __name__ == '__main__':
    TEMPERATURE_NET.load('static/model')
    PRECIPITATION_NET.load('static/model')
    # APP.run()
    webview.start(icon='static/assets/icon.png')
