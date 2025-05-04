from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

from src.synthetics import create_synthetic_data
from src.synthetics.create_associations import inventory_to_stations


def run_simulation(i):
    DURATION = 2  # seconds
    OUT_DIR = Path('data/test')
    N_CATALOGS = 1000
    AVG_RATE = i[0]
    RANGE = 0.1
    NOISE_PICKS = True
    PC_NOISE_PICKS = i[1]
    NOISE_TT = i[2]
    NOISE_GMV = i[3]
    MAX_MAGNITUDE = i[4]

    startdate = datetime(2025, 1, 1, 0, 0, 0)
    avg_events = DURATION * AVG_RATE
    min_events = int(avg_events - (avg_events * RANGE / 2))
    max_events = int(avg_events + (avg_events * RANGE / 2))

    stations = inventory_to_stations('stations/station_cords_blab_VALTER.csv')

    create_synthetic_data(
        OUT_DIR,
        N_CATALOGS,
        min_events,
        max_events,
        DURATION,
        stations,
        startdate=startdate,
        add_noise_picks=NOISE_PICKS,
        pc_noise_picks=PC_NOISE_PICKS,
        max_magnitude=MAX_MAGNITUDE,
        overwrite=False,
        noise_tt=NOISE_TT,
        noise_gmv=NOISE_GMV
    )


if __name__ == '__main__':
    setting = [
        (25, 0.1, 0.1, 0.05, -2.0),
        (25, 0.1, 0.1, 0.05, -1.0),
        (25, 0.1, 0.1, 0.05, 0.0),
        (25, 0.1, 0.1, 0.05, 1.0),
        (25, 0.1, 0.1, 0.05, 2.0),
        (50, 0.1, 0.1, 0.05, -2.0),
        (50, 0.1, 0.1, 0.05, -1.0),
        (50, 0.1, 0.1, 0.05, 0.0),
        (50, 0.1, 0.1, 0.05, 1.0),
        (50, 0.1, 0.1, 0.05, 2.0),
        (25, 0.2, 0.2,  0.1, -2.0),
        (25, 0.2, 0.2,  0.1, -1.0),
        (25, 0.2, 0.2,  0.1, 0.0),
        (25, 0.2, 0.2,  0.1, 1.0),
        (25, 0.2, 0.2,  0.1, 2.0),
        (50, 0.2, 0.2,  0.1, -2.0),
        (50, 0.2, 0.2,  0.1, -1.0),
        (50, 0.2, 0.2,  0.1, 0.0),
        (50, 0.2, 0.2,  0.1, 1.0),
        (50, 0.2, 0.2,  0.1, 2.0),
        (35, 0.15, 0.25, 0.15, -2.0),
        (35, 0.15, 0.25, 0.15, -1.0),
        (35, 0.15, 0.25, 0.15, 0.0),
        (35, 0.15, 0.25, 0.15, 1.0),
        (35, 0.15, 0.25, 0.15, 2.0)]

    with ProcessPoolExecutor() as executor:
        executor.map(run_simulation, setting)
