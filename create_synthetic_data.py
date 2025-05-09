from datetime import datetime
from pathlib import Path

from src.synthetics import create_synthetic_data
from src.synthetics.create_associations import inventory_to_stations

if __name__ == '__main__':

    for i in [15]:
        DURATION = 1  # seconds
        # OUT_DIR = Path(f'data/{DURATION}s_{i}hz')
        OUT_DIR = Path('data/test_30')
        N_CATALOGS = 10
        AVG_RATE = i  # events per second
        RANGE = 0.2  # percent around which to vary the number of events
        NOISE_PICKS = True
        PC_NOISE_PICKS = 1.0
        NOISE_TT = 0.1
        NOISE_GMV = 0.05
        MAX_MAGNITUDE = 0
        OVERWRITE = True

        startdate = datetime(2025, 1, 1, 0, 0, 0)
        avg_events = DURATION * AVG_RATE
        min_events = int(avg_events - (avg_events * RANGE/2))
        max_events = int(avg_events + (avg_events * RANGE/2))

        stations = inventory_to_stations(
            'stations/station_cords_blab_VALTER.csv')

        create_synthetic_data(OUT_DIR,
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
                              noise_gmv=NOISE_GMV)
