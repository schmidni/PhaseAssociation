import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

from src.synthetics import create_synthetic_data
from src.synthetics.create_associations import inventory_to_stations


def run_simulation(i):
    DURATION = i[1]  # seconds
    OUT_DIR = Path(f'data/test_{i[0]}_{i[2]}')
    N_CATALOGS = 20
    AVG_RATE = i[0]
    RANGE = 0.1
    NOISE_PICKS = True
    PC_NOISE_PICKS = i[2]
    NOISE_TT = i[3]
    NOISE_GMV = i[4]
    MAX_MAGNITUDE = i[5]
    OVERWRITE = True

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
        overwrite=OVERWRITE,
        noise_tt=NOISE_TT,
        noise_gmv=NOISE_GMV,
        id=i[6]
    )


if __name__ == '__main__':
    # avg_rate, pc_noise_picks, noise_tt, noise_gmv, max_magnitude, id
    setting = [
        (10, 1, 0.1, 0.1, 0.1, 0.0, 0),
        (10, 1, 0.3, 0.1, 0.1, 0.0, 1),
        (10, 1, 0.5, 0.1, 0.1, 0.0, 2),
        (10, 1, 0.7, 0.1, 0.1, 0.0, 4),
        (10, 1, 0.9, 0.1, 0.1, 0.0, 5),
    ]
    ncpu = multiprocessing.cpu_count() - 1
    with ProcessPoolExecutor(max_workers=ncpu) as executor:
        executor.map(run_simulation, setting)
