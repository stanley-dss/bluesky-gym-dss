import os
import random
import math

KORD_LAT = 41.9742
KORD_LON = -87.9073

TYPES = ["A320", "B738", "B744", "B757", "E190", "CRJ9"]

# picks a random lat/lon within a radius of a given location

def jitter_point(lat, lon, radius_km=5):
    r = radius_km * 1000 * random.random()
    theta = random.uniform(0, 2 * math.pi)
    dx = r * math.cos(theta)
    dy = r * math.sin(theta)
    dlat = dy / 111320
    dlon = dx / (111320 * math.cos(math.radians(lat)))
    return lat + dlat, lon + dlon

# calculates the heading degree between two lat/lon

def heading_deg(from_lat, from_lon, to_lat, to_lon):
    y = math.sin(math.radians(to_lon - from_lon)) * math.cos(math.radians(to_lat))
    x = (math.cos(math.radians(from_lat)) * math.sin(math.radians(to_lat)) -
         math.sin(math.radians(from_lat)) * math.cos(math.radians(to_lat)) *
         math.cos(math.radians(to_lon - from_lon)))
    return int((math.degrees(math.atan2(y, x)) + 360) % 360)

# converts feet to the altitude token used by the scenario format

def feet_to_alt_token(ft):
    if ft >= 18000:
        return f"FL{int(round(ft / 100))}"
    return str(int(round(ft)))

# FIXES represent the lat/lon of fixed waypoints that various airways go through

FIXES = {
    "W_GATE": (41.9742, -88.55),
    "W_MID":  (41.9742, -88.20),
    "CENTER": (41.9742, -87.9073),
    "E_MID":  (41.9742, -87.60),
    "E_GATE": (41.9742, -87.30),

    "N_GATE": (42.45, -87.9073),
    "N_MID":  (42.20, -87.9073),
    "S_MID":  (41.75, -87.9073),
    "S_GATE": (41.45, -87.9073),
}

# AIRWAYS describes the path (which of the FIXES are travelled through) of different airways 

AIRWAYS = {
    "WEST_TO_EAST": ["W_GATE", "W_MID", "CENTER", "E_MID", "E_GATE"],
    "EAST_TO_WEST": ["E_GATE", "E_MID", "CENTER", "W_MID", "W_GATE"],
    "NORTH_TO_SOUTH": ["N_GATE", "N_MID", "CENTER", "S_MID", "S_GATE"],
    "SOUTH_TO_NORTH": ["S_GATE", "S_MID", "CENTER", "N_MID", "N_GATE"],
}

# Altitude ranges (in feet) for each airway

AIRWAY_ALT_FT = {
    "WEST_TO_EAST": (26000, 34000),
    "EAST_TO_WEST": (28000, 36000),
    "NORTH_TO_SOUTH": (24000, 32000),
    "SOUTH_TO_NORTH": (30000, 38000),
}

def make_file(path, n_planes=28):
    lines = [
        "00:00:00.00>TRAILS ON",
        "00:00:00.00>RESO OFF",
        "00:00:00.00>RTF 10",
        "00:00:00.00>SWRAD LABEL",
        "00:00:00.00>PAN 41.97,-87.91",
        "#",
    ]

    callsigns = []


    for i in range(n_planes):
        cs = f"KL{104+i}"
        callsigns.append(cs)

        ac_type = random.choice(TYPES)

        # Start the aircraft near the airport (leaving some variance given different runways)
        lat, lon = jitter_point(KORD_LAT, KORD_LON, radius_km=8)

        # Pick an airway to follow upon departure
        airway_name = random.choice(list(AIRWAYS.keys()))
        airway = AIRWAYS[airway_name]

        first_lat, first_lon = FIXES[airway[0]]
        hdg = heading_deg(lat, lon, first_lat, first_lon)

        lo_ft, hi_ft = AIRWAY_ALT_FT[airway_name]
        alt_ft = random.randrange(lo_ft, hi_ft + 1, 1000)
        alt = feet_to_alt_token(alt_ft)
        
        spd = random.randint(250, 450)

        lines.append(f"# {cs} via {airway_name}")
        lines.append(
            f"00:00:00.00>CRE {cs},{ac_type},{lat:.6f},{lon:.6f},{hdg},{alt},{spd}"
        )

        for fix in airway:
            wlat, wlon = FIXES[fix]
            lines.append(f"00:00:00.00>{cs} ADDWPT {wlat:.6f},{wlon:.6f}")


        end_lat, end_lon = FIXES[airway[-1]]
        for _ in range(random.randint(0, 2)):
            wlat, wlon = jitter_point(end_lat, end_lon, radius_km=20)
            lines.append(f"00:00:00.00>{cs} ADDWPT {wlat:.6f},{wlon:.6f}")

    lines.append("")
    for cs in reversed(callsigns):
        lines.append(f"00:06:40.00>DEL {cs}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

def main():
    random.seed(0)
    os.makedirs("scenarios_kord", exist_ok=True)

    for k in range(1, 5000):
        make_file(f"scenarios_kord/scenario_{k}.scn")

    print("Made scenarios: all aircraft start near KORD")

if __name__ == "__main__":
    main()