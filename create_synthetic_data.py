from pathlib import Path

from src.synthetics import create_synthetic_data
from src.synthetics.create_associations import inventory_to_stations

if __name__ == '__main__':

    for i in [0.1, 0.5, 1, 2, 3, 5, 8, 13, 18, 22,
              26, 30, 35, 40, 50]:
        OUT_DIR = Path(f'data/reference/30s_{i}hz')
        N_CATALOGS = 5000
        DURATION = 30  # seconds
        AVG_RATE = i  # events per second
        RANGE = 0.2  # percent around which to vary the number of events
        NOISE_PICKS = True

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
                              add_noise_picks=NOISE_PICKS,
                              overwrite=True)
