from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

from src.synthetics import create_synthetic_data
from src.synthetics.create_associations import inventory_to_stations


def run_simulation(i):
    DURATION = 2  # seconds
    OUT_DIR = Path('data/test')
    N_CATALOGS = 1500
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
        noise_gmv=NOISE_GMV,
        id=i[5]
    )


if __name__ == '__main__':
    setting = [
        (25, 0.1, 0.1, 0.05, -2.0, 0),
        (25, 0.1, 0.1, 0.05, -1.0, 1),
        (25, 0.1, 0.1, 0.05, 0.0, 2),
        (25, 0.1, 0.1, 0.05, 1.0, 3),
        (25, 0.1, 0.1, 0.05, 2.0, 4),
        (50, 0.1, 0.1, 0.05, -2.0, 5),
        (50, 0.1, 0.1, 0.05, -1.0, 6),
        (50, 0.1, 0.1, 0.05, 0.0, 7),
        (50, 0.1, 0.1, 0.05, 1.0, 8),
        (50, 0.1, 0.1, 0.05, 2.0, 9),
        (25, 0.2, 0.2,  0.1, -2.0, 10),
        (25, 0.2, 0.2,  0.1, -1.0, 11),
        (25, 0.2, 0.2,  0.1, 0.0, 12),
        (25, 0.2, 0.2,  0.1, 1.0, 13),
        (25, 0.2, 0.2,  0.1, 2.0, 14),
        (50, 0.2, 0.2,  0.1, -2.0, 15),
        (50, 0.2, 0.2,  0.1, -1.0, 16),
        (50, 0.2, 0.2,  0.1, 0.0, 17),
        (50, 0.2, 0.2,  0.1, 1.0, 18),
        (50, 0.2, 0.2,  0.1, 2.0, 19),
        (35, 0.15, 0.25, 0.15, -2.0, 20),
        (35, 0.15, 0.25, 0.15, -1.0, 21),
        (35, 0.15, 0.25, 0.15, 0.0, 22),
        (35, 0.15, 0.25, 0.15, 1.0, 23),
        (35, 0.15, 0.25, 0.15, 2.0, 24),
    ]

    with ProcessPoolExecutor(max_workers=12) as executor:
        executor.map(run_simulation, setting)
