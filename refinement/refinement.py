# The MIT License (MIT)
#
# Copyright (c) 2023 Yifeng Lyu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
    
import math
import networkx as nx
from multiprocessing import Process
import multiprocessing
import datetime
import sys
import os


sys.path.append(os.path.abspath(os.path.dirname(os.getcwd())) + "/initialization")
from initialization import util
import numpy as np
from multiprocessing import Process, Pipe
import copy
import os
from collections import defaultdict
import time
import random


new_graph = nx.Graph() 
 
config = sys.argv[1] 
number_of_pipes=20
E_before_delete = 1
E_after_delete = 1 

parent_pipes = []
child_pipes = []
parent_pipes_1 = []
child_pipes_1 = []

for i in range(number_of_pipes):
    parent_pipe, child_pipe = Pipe() 
    parent_pipes.append(parent_pipe)
    child_pipes.append(child_pipe)
for i in range(number_of_pipes):
    parent_pipe, child_pipe = Pipe() 
    parent_pipes_1.append(parent_pipe)
    child_pipes_1.append(child_pipe)

satPositionsFile = "../input/constellation_" + config + "/data_sat_position/sat_positions_0.txt"
validISLFile = "../input/constellation_" + config + "/data_validISLs/valid_ISLs_0.txt"
cityPositionsFile = "../input/data_cities/cities.txt"
cityCoverageFile = "../input/constellation_" + config + "/data_coverage/city_coverage_0.txt"
cityPairFile = "../input/data_cities/city_pairs_rand_5K.txt"
EARTH_RADIUS = 6371
NUM_CITY_PAIRS = 5000
city_pairs_each_process = int(NUM_CITY_PAIRS/number_of_pipes)

default_weight = 10000000

if config == "starlink":
    NUM_ORBITS = 72
    NUM_SATS_PER_ORBIT = 22
    TOTAL_NUMBER_SATS = 1584 
    ACTION_DIM = 167688     
elif config == "oneweb":
    NUM_ORBITS = 18
    NUM_SATS_PER_ORBIT = 40
    TOTAL_NUMBER_SATS = 720
    ACTION_DIM = 77337     
elif config == "telesat":
    NUM_ORBITS = 27
    NUM_SATS_PER_ORBIT = 13
    TOTAL_NUMBER_SATS = 351 
    ACTION_DIM = 16001               
else:
    print("Unable to identify configuration: "+config)
    exit(1)
city_positions = {}
city_positions, new_graph = util.read_city_positions(cityPositionsFile, new_graph)

array_Q = np.zeros((NUM_ORBITS * NUM_SATS_PER_ORBIT, NUM_ORBITS * NUM_SATS_PER_ORBIT))
array_U = np.zeros((NUM_ORBITS * NUM_SATS_PER_ORBIT, NUM_ORBITS * NUM_SATS_PER_ORBIT))

###############------------------------starlink----------------------------######################
#initialized_file = '../output/starlink/100.txt'  #      

###############------------------------oneweb----------------------------######################
#initialized_file = '../output/oneweb/100.txt'

###############------------------------telesat----------------------------######################
#initialized_file = '../output/telesat/100.txt' 

sat_positions = {}
city_coverage = {}

city_pairs = {}
city_pairs = util.read_city_pair_file(cityPairFile)

test_dynamic_number=1  

def write_edges_to_file(graph, writer):
    edges = graph.edges(data=True)
    for edge in edges:
        writer.write(str(edge[0]) + "," + str(edge[1]) + "\n")

def read_sat_positions(sat_pos_file):
    global sat_positions
    sat_positions = {}
    lines = [line.rstrip('\n') for line in open(sat_pos_file)]
    for i in range(len(lines)):
        val = lines[i].split(",")
        sat_positions[int(val[0])] = {
            "orb_id": int(val[1]),
            "orb_sat_id": int(val[2]),
            "lat_deg": float(val[3]),
            "lat_rad": math.radians(float(val[3])),
            "long_deg": float(val[4]),
            "long_rad": math.radians(float(val[4])),
            "alt_km": float(val[5])
        }

def proc_send(grph, pipe, signals, city_coverage):
    weightSum = 0
    weightedDistanceSum = 0
    inverse_weightedDistanceSum = 0
    weightedHopCountSum = 0
    weightedrtt = 0
    weightedpower = 0
    betweeness_board = np.zeros((TOTAL_NUMBER_SATS, TOTAL_NUMBER_SATS))

    new_graph = copy.deepcopy(grph)
    power_for_total = 0
    CDF_RTT_part = np.zeros(city_pairs_each_process, )
    betweeness_dict = {}
    
    for u in range(new_graph.number_of_nodes()):
        for v in range(new_graph.number_of_nodes()):
            if new_graph.has_edge(u, v):
                new_graph.edges[u, v]['length'] = util.compute_isl_length(u, v, sat_positions)

    for i in range(int(len(city_pairs) / number_of_pipes) * signals,
                   int(len(city_pairs) / number_of_pipes) * (signals + 1)):
        city1 = city_pairs[i]["city_1"]
        city2 = city_pairs[i]["city_2"]
        util.add_coverage_for_city(new_graph, city1, city_coverage)
        util.add_coverage_for_city(new_graph, city2, city_coverage)
        distance_lists = []
        inverse_distance_lists = []
        hops_lists = []
        power_lists = []
        power_for_each_pair = 0
        deleted_links_1 = []
        deleted_links_2 = []
        deleted_links_3 = []
        power_lists = []
        distance_sum = 0
        inverse_distance_sum = 0
        hops_sum = 0
        power_sum = 0
        disconnection_flag = 0

        for j in range(1):

            if nx.has_path(new_graph, source=city1, target=city2):
                disconnection_flag = 0
                path = nx.shortest_path(new_graph, source=city1, target=city2, weight='length')
                distance = nx.shortest_path_length(new_graph, source=city1, target=city2, weight='length')

                for j in range(path.__len__()-3):

                    if str(path[j+1]) + "_" + str(path[j+2]) in betweeness_dict:
                        betweeness_dict[ str(path[j+1]) + "_" + str(path[j+2]) ] += 1
                    else:
                        betweeness_dict[ str(path[j+1]) + "_" + str(path[j+2]) ] = 1

                hops = path.__len__() - 1
                inverse_distance = 1 / distance 
                distance_lists.append(distance)
                inverse_distance_lists.append(inverse_distance)
                hops_lists.append(hops)
                for x in range(hops):
                    power_loss =  32.4 + 20 * math.log(( new_graph.edges[path[x], path[x + 1]]['length'] * 13.5 * 1000), 10)      
                    power_for_each_hop_in_db = -158.5 + power_loss
                    power_for_each_hop_in_w = 10 ** (power_for_each_hop_in_db/10)                   
                    power_for_each_pair += power_for_each_hop_in_w
                power_lists.append(power_for_each_pair)    
                for x in range(hops - 2):
                    deleted_links_1.append(path[x + 1])
                    deleted_links_2.append(path[x + 2])
                    deleted_links_3.append(new_graph.edges[path[x + 1], path[x + 2]]['length'])
                    new_graph.remove_edge(path[x + 1], path[x + 2])
            else:
                disconnection_flag = 1
                util.remove_coverage_for_city(new_graph, city1, city_coverage)
                util.remove_coverage_for_city(new_graph, city2, city_coverage)
                new_graph.remove_node(city1)
                new_graph.remove_node(city2)
        if disconnection_flag == 0: 
            for k in range(len(deleted_links_1)):
                new_graph.add_edge(deleted_links_1[k], deleted_links_2[k], length=util.compute_isl_length(int(deleted_links_1[k]), int(deleted_links_2[k]), sat_positions))
            for j in range(len(distance_lists)):
                distance_sum += distance_lists[j]
                inverse_distance_sum += inverse_distance_lists[j]
                hops_sum += hops_lists[j]
                power_sum += power_lists[j]
            average_distance = distance_sum / len(distance_lists)
            inverse_average_distance = inverse_distance_sum / len(inverse_distance_lists)
            average_hops = hops_sum / len(distance_lists)
            average_power = power_sum / len(power_lists)

            average_rtt = average_distance/300000*1000 + (average_hops - 1) * 0.1 
            CDF_RTT_part[i%city_pairs_each_process] = average_rtt

        else:
            average_distance = 0
            average_hops = 0
            average_rtt = 0
            inverse_average_distance = 0
            average_power = 0
        weight = city_positions[city1]["pop"] * city_positions[city2]["pop"] / default_weight
        weightSum += weight
        weightedDistanceSum += average_distance * weight
        inverse_weightedDistanceSum += inverse_average_distance * weight
        weightedHopCountSum += average_hops * weight
        weightedrtt += average_rtt * weight
        weightedpower += average_power * weight
        util.remove_coverage_for_city(new_graph, city1, city_coverage)
        util.remove_coverage_for_city(new_graph, city2, city_coverage)
        if new_graph.has_node(city1):
            new_graph.remove_node(city1)
        if new_graph.has_node(city2):            
            new_graph.remove_node(city2) 
    result = []
    result.append(os.getpid())
    result.append(weightSum)
    result.append(weightedDistanceSum)
    result.append(weightedHopCountSum)
    result.append(weightedrtt)
    result.append(inverse_weightedDistanceSum)
    result.append(weightedpower)
    result.append(CDF_RTT_part) 
    result.append(betweeness_dict)  
    pipe.send(result)
    return

def proc_send_1(grph, pipe, signals):
    city_coverage = util.read_city_coverage(cityCoverageFile)
    weightSum = 0
    weightedDistanceSum = 0
    weightedHopCountSum = 0
    weightedRttSum = 0
    betweeness_board = np.zeros((TOTAL_NUMBER_SATS, TOTAL_NUMBER_SATS))
    grph = copy.deepcopy(grph)
    for i in range(int(NUM_CITY_PAIRS / number_of_pipes) * signals,
                   int(NUM_CITY_PAIRS / number_of_pipes) * (signals + 1)): 
        city1 = city_pairs[i]["city_1"]
        city2 = city_pairs[i]["city_2"]
        util.add_coverage_for_city(grph, city1, city_coverage)
        util.add_coverage_for_city(grph, city2, city_coverage)
        distance_lists = []
        hops_lists = []
        deleted_links_1 = [] 
        deleted_links_2 = [] 
        deleted_links_3 = []            
        distance_sum = 0
        hops_sum = 0             
        if nx.has_path(grph, source=city1, target=city2):
            for j in range(1):          
                if nx.has_path(grph, source=city1, target=city2):
                    distance = nx.shortest_path_length(grph, source=city1, target=city2, weight='length')
                    path = nx.shortest_path(grph, source=city1, target=city2, weight='length')
                    hops = path.__len__() - 1    
                    distance_lists.append(distance)
                    hops_lists.append(hops)
                    for x in range(hops - 2):
                        deleted_links_1.append(path[x+1])
                        deleted_links_2.append(path[x+2])
                        deleted_links_3.append(grph.edges[path[x+1], path[x+2]]['length']) 
                        betweeness_board[path[x+1]][path[x+2]] += city_positions[city1]["pop"] * city_positions[city2]["pop"] / default_weight
                        betweeness_board[path[x+2]][path[x+1]] += city_positions[city1]["pop"] * city_positions[city2]["pop"] / default_weight                         
                        grph.remove_edge(path[x+1], path[x+2])  
                else: 
                    break
            for k in range(len(deleted_links_1)):            
                grph.add_edge(deleted_links_1[k], deleted_links_2[k], length=util.compute_isl_length(int(deleted_links_1[k]), int(deleted_links_2[k]), sat_positions))            
            for j in range(len(distance_lists)): 
                distance_sum += distance_lists[j] 
                hops_sum += hops_lists[j]        
            average_distance = distance_sum/len(distance_lists)        
            average_hops = hops_sum/len(distance_lists)        
            weight = city_positions[city1]["pop"] * city_positions[city2]["pop"] / default_weight
            weightSum += weight
            weightedDistanceSum += average_distance * weight
            weightedHopCountSum += average_hops * weight        
            util.remove_coverage_for_city(grph, city1, city_coverage)
            util.remove_coverage_for_city(grph, city2, city_coverage)
            grph.remove_node(city1)
            grph.remove_node(city2) 
        else:
            util.remove_coverage_for_city(grph, city1, city_coverage)
            util.remove_coverage_for_city(grph, city2, city_coverage)
            grph.remove_node(city1)
            grph.remove_node(city2) 
            result = []
            result.append(os.getpid())
            result.append(weightSum)
            result.append(weightedDistanceSum)
            result.append(weightedHopCountSum)
            betweeness_board = betweeness_board.reshape(TOTAL_NUMBER_SATS * TOTAL_NUMBER_SATS, 1)
            result.append(betweeness_board)             
            pipe.send(result)  
            return
    result = []
    result.append(os.getpid())
    result.append(weightSum)
    result.append(weightedDistanceSum)
    result.append(weightedHopCountSum)
    betweeness_board = betweeness_board.reshape(TOTAL_NUMBER_SATS * TOTAL_NUMBER_SATS, 1)
    result.append(betweeness_board)    
    pipe.send(result)            

def compute_metric(graph, order):
    graph_temp = copy.deepcopy(graph)
    final_Distance_Sum = 0
    inverse_final_Distance_Sum = 0
    final_Hop_Sum = 0
    final_Rtt_Sum = 0
    final_Power_Sum = 0
    CDF_RTT = np.zeros(NUM_CITY_PAIRS, )  
    Betweeness_Dict = {} 
    Betweeness_array = np.zeros(graph_temp.number_of_edges()*2, ) 
    global satPositions_lists
    global E_before_delete
    global E_after_delete
    
    satPositions_lists = []
    read_sat_positions(satPositionsFile)
    city_coverage = util.read_city_coverage(cityCoverageFile)
    satPositions_lists.append(sat_positions)
    weightSum = 0
    weightedDistanceSum = 0
    inverse_weightedDistanceSum = 0
    weightedHopCountSum = 0
    weightedRttSum = 0
    weightsum_total = 0
    Distance_total = 0
    inverse_Distance_total = 0
    Hop_total = 0
    Rtt_total = 0
    Power_total = 0
    betweeness_vec = np.zeros(graph_temp.number_of_edges(), )
    proce_list = []
    for i in range(number_of_pipes):
        pro = Process(target=proc_send, args=(graph_temp, child_pipes[i], i, city_coverage))
        proce_list.append(pro)
        proce_list[i].start()
    for i in range(number_of_pipes):
        idd, weightsum, distance, hop, rtt, inverse_distance, power, CDF_RTT_part, betweeness_dict_part = parent_pipes[i].recv()  
        weightsum_total += weightsum
        Distance_total += distance
        inverse_Distance_total += inverse_distance
        Hop_total += hop
        Rtt_total += rtt
        Power_total += power
        CDF_RTT[i*city_pairs_each_process:i*city_pairs_each_process+city_pairs_each_process] = CDF_RTT_part
        for key,value in betweeness_dict_part.items():
            if key in Betweeness_Dict:
                Betweeness_Dict[ key ] += 1
            else:
                Betweeness_Dict[ key ] = 0            
    between_index = 0
    for key,value in Betweeness_Dict.items():
        Betweeness_array[between_index] = value        
        between_index += 1
    sorted_Betweeness_array = -np.sort(-Betweeness_array)      
    Distance_total = Distance_total / weightsum_total
    inverse_Distance_total = inverse_Distance_total / weightsum_total
    Hop_total = Hop_total / weightsum_total
    Rtt_total = Rtt_total / weightsum_total
    Power_total = Power_total / weightsum_total
    for i in range(number_of_pipes):
        proce_list[i].join()
    final_Distance_Sum += Distance_total
    inverse_final_Distance_Sum += inverse_Distance_total
    final_Hop_Sum += Hop_total
    final_Rtt_Sum += Rtt_total
    final_Power_Sum += Power_total 
    final_Distance_Sum = final_Distance_Sum
    inverse_final_Distance_Sum = inverse_final_Distance_Sum 
    if order == 0:
        E_before_delete = inverse_final_Distance_Sum
    else:
        E_after_delete = inverse_final_Distance_Sum    
    final_Hop_Sum = final_Hop_Sum    
    final_Rtt_Sum = final_Rtt_Sum
    final_Power_Sum = final_Power_Sum
    Power_total_in_w = final_Power_Sum
    final_wMetric = final_Rtt_Sum   
    result = " "
    if order == 0:
        result = str(final_Distance_Sum) + " " + str(final_Hop_Sum) + " " + str(final_wMetric) + " " + str(Power_total_in_w)  
    return result, final_wMetric
          
def compute_metric_and_robustness(graph):
    graph_temp = copy.deepcopy(graph)
    weightSum = 0
    weightedDistanceSum = 0
    weightedHopCountSum = 0
    weightedRttSum = 0
    global E_after_delete
    global E_before_delete

    proce_list = []
    for i in range(number_of_pipes):
        pro = Process(target=proc_send_1, args=(graph_temp, child_pipes_1[i], i))
        proce_list.append(pro)
        proce_list[i].start()

    weightsum_total = 0
    Distance_total = 0
    Hop_total = 0
    Total_betweeness_board = np.zeros((TOTAL_NUMBER_SATS, TOTAL_NUMBER_SATS))
    
    for i in range(number_of_pipes):
        idd, weightsum, distance, hop, betweeness_board = parent_pipes_1[i].recv() 
        weightsum_total += weightsum
        Distance_total += distance
        Hop_total += hop
        betweeness_board = betweeness_board.reshape((TOTAL_NUMBER_SATS, TOTAL_NUMBER_SATS))
        Total_betweeness_board += betweeness_board

    for i in range(number_of_pipes):
        proce_list[i].join()    
    Distance_total = Distance_total / weightsum_total
    Hop_total = Hop_total / weightsum_total

    betweeness_vec = np.zeros(graph_temp.number_of_edges(), )

    robust_list_x = []
    robust_list_y = []
    index = 0
    
    if config == "starlink":
        highest_number_of_robust_nodes_10 = 150
    elif config == "oneweb":
        highest_number_of_robust_nodes_10 = 70   
    elif config == "telesat":
        highest_number_of_robust_nodes_10 = 35     
    
    max_robust = 0
    delete_node_x = 0
    delete_node_y = 0
    
    #=============remove highest nodes=============    
    copy_Total_betweeness_board = copy.deepcopy(Total_betweeness_board) 
    robust_node_lists = np.zeros(TOTAL_NUMBER_SATS)
    robust_node_lists = np.sum(copy_Total_betweeness_board, axis = 1)
    for i in range(highest_number_of_robust_nodes_10):
        max_robust = 0
        for x in range(TOTAL_NUMBER_SATS): 
            if robust_node_lists[x] > max_robust:
                max_robust = robust_node_lists[x]
                delete_node_x = x
        robust_list_x.append(delete_node_x)
        robust_node_lists[delete_node_x] = 0
    delay_result, delay_num = compute_metric(graph_temp, 0)    
    for i in range(highest_number_of_robust_nodes_10):    
        if graph_temp.has_node(robust_list_x[i]):
            graph_temp.remove_node(robust_list_x[i])   # highest nodes          
    compute_metric(graph_temp, 1)   
    ratio = E_after_delete/E_before_delete
    return ratio, delay_num, delay_result

def edge_switching(graph):
    a_index = random.randint(1, graph.number_of_nodes()-1)
    b_list = list(graph.neighbors(a_index))
    if not b_list:
        return False, -1, -1, -1, -1
    array_temp = array_Q + array_U
    min_b = b_list[0]
    for i in range(len(b_list)):
        if array_temp[a_index][b_list[i]] < array_temp[a_index][min_b]:
            min_b = b_list[i]
    b_index = min_b
    graph.remove_edge(a_index, b_index)#+ 
    
    for i in range(len(valid_isls)):
        if valid_isls[i]["sat_1"] == a_index and not graph.has_edge(a_index, valid_isls[i]["sat_2"]):    
            max_c = valid_isls[i]["sat_2"]
        else:
            dist = util.compute_isl_length(a_index, b_index, sat_positions)
            graph.add_edge(a_index, b_index, length = dist) 
            return False, -1, -1, -1, -1
    for i in range(len(valid_isls)):
        if valid_isls[i]["sat_1"] == a_index and not graph.has_edge(a_index, valid_isls[i]["sat_2"]) and array_temp[a_index][valid_isls[i]["sat_2"]] > array_temp[a_index][max_c]:
            max_c = valid_isls[i]["sat_2"]
     
        if valid_isls[i]["sat_2"] == a_index and not graph.has_edge(a_index, valid_isls[i]["sat_1"]) and array_temp[a_index][valid_isls[i]["sat_1"]] > array_temp[a_index][max_c]:
            max_c = valid_isls[i]["sat_1"]

    c_index = max_c
    flag_1 = 0
    for i in range(len(valid_isls)):
        if (valid_isls[i]["sat_1"] == a_index and valid_isls[i]["sat_2"] == c_index) or (valid_isls[i]["sat_1"] == c_index and valid_isls[i]["sat_2"] == a_index):    
            dist = util.compute_isl_length(a_index, c_index, sat_positions)
            graph.add_edge(a_index, c_index, length = dist) #+ 
            flag_1 = 1   
            break
    if flag_1 == 0:
        dist = util.compute_isl_length(a_index, b_index, sat_positions)
        graph.add_edge(a_index, b_index, length = dist) 
        graph.remove_edge(a_index, c_index)
        return False, -1, -1, -1, -1
    
    d_list = list(graph.neighbors(c_index))
    min_d = d_list[0]
    for i in range(len(d_list)):  
        if i == a_index:
            continue
        if array_temp[c_index][d_list[i]] < array_temp[c_index][min_d]:
            min_d = d_list[i]
    d_index = min_d    
    graph.remove_edge(c_index, d_index) #+ 
    flag_2 = 0
    for i in range(len(valid_isls)):
        if (valid_isls[i]["sat_1"] == b_index and valid_isls[i]["sat_2"] == d_index) or (valid_isls[i]["sat_1"] == d_index and valid_isls[i]["sat_2"] == b_index):    
            dist = util.compute_isl_length(b_index, d_index, sat_positions)
            graph.add_edge(b_index, d_index, length = dist) #+    
            flag_2 = 1
            break
    if flag_2 == 0:
        dist = util.compute_isl_length(a_index, b_index, sat_positions)
        graph.add_edge(a_index, b_index, length = dist) 
        graph.remove_edge(a_index, c_index)
        dist = util.compute_isl_length(c_index, d_index, sat_positions)
        graph.add_edge(c_index, d_index, length = dist)
        graph.remove_edge(b_index, d_index)        
        return False, -1, -1, -1, -1
  
    return True, a_index, b_index, c_index, d_index

edge_switching_result = "../output/" + config + "/test_result.txt"          

for j in range(TOTAL_NUMBER_SATS):
    new_graph.add_node(j)
lines = [line.rstrip('\n') for line in open(initialized_file)]
read_sat_positions(satPositionsFile)
for i in range(len(lines)):
    val = lines[i].split(",")
    dist = util.compute_isl_length(int(val[0]), int(val[1]), sat_positions)
    new_graph.add_edge(int(val[0]), int(val[1]), length=dist)
       
valid_isls = util.read_valid_isls(validISLFile)

before_switching_robust_result = 0
before_switching_delay_result = 0
after_switching_robust_result = 0
after_switching_delay_result = 0
real_episode = 0

while real_episode < 100:
    robust_result, delay_result, before_all_result = compute_metric_and_robustness(new_graph)
    before_switching_robust_result = robust_result
    before_switching_delay_result = delay_result
    
    store_graph = copy.deepcopy(new_graph)
    switch_result, a_index, b_index, c_index, d_index = edge_switching(new_graph)
    if switch_result == False:
        continue
       
    robust_result, delay_result, after_all_result = compute_metric_and_robustness(new_graph)
    after_switching_robust_result = robust_result
    after_switching_delay_result = delay_result  
    delta_delay = before_switching_delay_result - after_switching_delay_result
    delta_robust = before_switching_robust_result - after_switching_robust_result
    if (before_switching_robust_result < after_switching_robust_result) and (before_switching_delay_result > after_switching_delay_result) : 
        real_episode += 1  
        f = open(edge_switching_result, "a+")
        f.write(after_all_result + " " + str(after_switching_robust_result) + "\n")
        episode_overall = "../output/" + config + "/episode_" + str(real_episode) + ".txt" 
        best_file = open(episode_overall, 'a+')
        write_edges_to_file(new_graph, best_file)
        
        reward = 0.5 * delta_delay - 0.5 * 100 * delta_robust
        array_Q[a_index][b_index] -= reward
        array_Q[c_index][d_index] -= reward
        array_Q[a_index][c_index] += reward
        array_Q[b_index][d_index] += reward    

        array_Q[b_index][a_index] -= reward
        array_Q[d_index][c_index] -= reward
        array_Q[c_index][a_index] += reward
        array_Q[d_index][b_index] += reward                
        
    else:
        new_graph = store_graph

    array_U[a_index][b_index] += 1
    array_U[a_index][c_index] += 1
    array_U[c_index][d_index] += 1
    array_U[b_index][d_index] += 1
    
    array_U[b_index][a_index] += 1
    array_U[c_index][a_index] += 1
    array_U[d_index][c_index] += 1
    array_U[d_index][b_index] += 1



