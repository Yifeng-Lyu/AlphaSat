import math
EARTH_RADIUS = 6371  


def compute_isl_length(sat1, sat2, sat_positions):
    x1 = (EARTH_RADIUS + sat_positions[sat1]["alt_km"]) * math.cos(sat_positions[sat1]["lat_rad"]) * math.sin(
        sat_positions[sat1]["long_rad"])
    y1 = (EARTH_RADIUS + sat_positions[sat1]["alt_km"]) * math.sin(sat_positions[sat1]["lat_rad"])
    z1 = (EARTH_RADIUS + sat_positions[sat1]["alt_km"]) * math.cos(sat_positions[sat1]["lat_rad"]) * math.cos(
        sat_positions[sat1]["long_rad"])
    x2 = (EARTH_RADIUS + sat_positions[sat2]["alt_km"]) * math.cos(sat_positions[sat2]["lat_rad"]) * math.sin(
        sat_positions[sat2]["long_rad"])
    y2 = (EARTH_RADIUS + sat_positions[sat2]["alt_km"]) * math.sin(sat_positions[sat2]["lat_rad"])
    z2 = (EARTH_RADIUS + sat_positions[sat2]["alt_km"]) * math.cos(sat_positions[sat2]["lat_rad"]) * math.cos(
        sat_positions[sat2]["long_rad"])
    dist = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2) + math.pow((z2 - z1), 2))
    return dist

def read_valid_isls(valid_isl_file):
    valid_isls = {}
    lines = [line.rstrip('\n') for line in open(valid_isl_file)]
    for i in range(len(lines)):
        val = lines[i].split(",")
        valid_isls[i] = {
            "sat_1": int(val[0]),
            "sat_2": int(val[1]),
            "dist_km": float(val[2])
        }
    return valid_isls

def read_city_positions(city_pos_file, graph):
    city_positions = {}
    lines = [line.rstrip('\n') for line in open(city_pos_file)]
    for i in range(len(lines)):
        val = lines[i].split(",")
        city_positions[int(val[0])] = {
            "lat_deg": float(val[2]),
            "long_deg": float(val[3]),
            "pop": float(val[4])
        }
    return city_positions, graph

def read_city_coverage(coverage_file):
    city_coverage = {}
    lines = [line.rstrip('\n') for line in open(coverage_file)]
    for i in range(len(lines)):
        val = lines[i].split(",")
        city_coverage[i] = {
            "city": int(val[0]),
            "sat": int(val[1]),
            "dist": float(val[2])
        }
    return city_coverage

def read_city_pair_file(city_pair_file):
    city_pairs = {}
    lines = [line.rstrip('\n') for line in open(city_pair_file)]
    for i in range(len(lines)):
        val = lines[i].split(",")
        city_pairs[i] = {
            "city_1": int(val[0]),
            "city_2": int(val[1]),
            "geo_dist": float(val[2])
        }
    return city_pairs


def add_coverage_for_city(graph, city, city_coverage):
    for i in range(len(city_coverage)):
        if city_coverage[i]["city"] == city:
            graph.add_edge(city_coverage[i]["city"], city_coverage[i]["sat"], length=city_coverage[i]["dist"])
    return graph

def remove_coverage_for_city(graph, city, city_coverage):
    for i in range(len(city_coverage)):
        if city_coverage[i]["city"] == city and graph.has_edge(city_coverage[i]["city"], city_coverage[i]["sat"]):
            graph.remove_edge(city_coverage[i]["city"], city_coverage[i]["sat"])
    return graph

