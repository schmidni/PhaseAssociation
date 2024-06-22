import csv
import sys
from urllib.parse import urlparse

from obspy.clients.fdsn import Client

from tools.coordinates import createBULGGTransformer


def create_FDSN_client(url):
    parsed = urlparse(url)
    base_url = parsed._replace(
        netloc=f"{parsed.hostname}:{parsed.port}").geturl()
    if parsed.username and parsed.password:
        print(f"Connecting to {base_url} with credentials", file=sys.stderr)
        return Client(base_url=base_url,
                      user=parsed.username,
                      password=parsed.password)
    else:
        print(f"Connecting to {base_url}", file=sys.stderr)
        return Client(base_url=base_url)


def main():
    starttime = "2023-09-10T11:00:00"
    endtime = "2023-09-10T12:00:00"
    client = create_FDSN_client(
        "http://user:BedrettoLab2017!@bedretto-dev2.ethz.ch:8080")
    inventory = client.get_stations(network="8R",
                                    station="V*",
                                    location="*",
                                    channel="J*",
                                    starttime=starttime,
                                    endtime=endtime,
                                    level="response")

    local_cords = createBULGGTransformer()
    # stations = {}
    with open(
        '../projects/functions/plotting/data/station_cords_blab_VALTER.csv',
            'w', newline='') as outcsv:
        writer = csv.writer(outcsv, delimiter=',')
        writer.writerow(["network_code", "station_code", "location_code",
                        "chanel_code", "longitude", "latitude", 'elevation',
                         'x', 'y', 'z'])

        for net in inventory.networks:
            for sta in net.stations:
                for cha in sta.channels:
                    x, y, z = local_cords.to_local_coords(
                        cha.longitude, cha.latitude, cha.elevation)
                    row = [f"{net.code}", f"{sta.code}",
                           f"{cha.location_code}", f"{cha.code}",
                           f"{cha.longitude:f}", f"{cha.latitude:f}",
                           f"{cha.elevation:f}",
                           f"{x:.3f}", f"{y:.3f}", f"{z:.3f}"]
                    writer.writerow(row)


if __name__ == '__main__':
    main()
