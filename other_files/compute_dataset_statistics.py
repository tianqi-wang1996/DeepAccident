import glob
import pdb
import random
import numpy as np
import re
# vehicle_name_list = ['ego_vehicle/', 'other_vehicle/', 'ego_vehicle_behind/', 'other_vehicle_behind/']
folder_name_list = ['type1_subtype1_accident', 'type1_subtype1_normal', 'type1_subtype2_accident', 'type1_subtype2_normal']


# folder_name_list = ['type1_subtype1_accident', 'type1_subtype2_accident']
# folder_name_list = ['type1_subtype1_normal', 'type1_subtype2_normal']

# folder_name = './data/carla_perception/meta/*'


file_list = []
for folder_name in folder_name_list:
    rootdir = './data/DeepAccident_data/' + folder_name + '/meta/*'
    file_list += sorted(glob.glob(rootdir))
# type1_exists = os.path.exists(rootdir)

# HardRainNoon 1236 car 1237 car 6790.168605131015 same front_side 78
#  colliding agents: ego_behind other_behind
#  agents id: 1223 1236 1212 1237
#  road_type: four-way junction
#  another_vehicle_spawn_side: left
#  ego_vehicle_direction: straight
#  other_vehicle_direction: straight

weather_name_list = []
crash_list = []
scenario_length_list = []
scenario_type_list = []
for file_single in file_list:
    with open(file_single) as f:
        lines = [line.rstrip('\n') for line in f]
        line = lines[0]
        scenario_length_list.append(int(line.split(' ')[-1])-10)
        weather_name_list.append(line.split(' ')[0])
        # if line.split(' ')[1] != '-1':
        #     print(file_single.split('/')[-1])
        crash_list.append(line.split(' ')[1] != '-1')
        subtype_name = file_single.split('/')[-1].split('_')[2]
        road_type = lines[3].split(': ')[-1]
        spawn_side = lines[4].split(': ')[-1]
        ego_direction = lines[5].split(': ')[-1]
        other_direction = lines[6].split(': ')[-1]

        accident_id1, accident_cls1, accident_id2, accident_cls2 = int(lines[0].split(' ')[1]), lines[0].split(' ')[2], int(lines[0].split(' ')[3]), lines[0].split(' ')[4]
        designed_ids_str = lines[2].split(': ')[-1].split(' ')
        designed_ids = [int(designed_ids_str[0]), int(designed_ids_str[1]), int(designed_ids_str[2]), int(designed_ids_str[3])]

        scenario_type = 'None'
        if accident_id1 == -1 or accident_id2 == -1:
            scenario_type = 'no accident'
        elif accident_cls1 == 'pedestrian' or accident_cls2 == 'pedestrian':
            scenario_type = 'collide with pedestrians'
        elif accident_id1 not in designed_ids or accident_id2 not in designed_ids:
            scenario_type = 'collide with other vehicles'
        else:
            if road_type == 'four-way junction':
                if ego_direction == 'straight' and other_direction == 'straight' and spawn_side != 'opposite':
                    scenario_type = subtype_name + ' ' + 'straight straight'
                elif spawn_side != 'opposite' and ((ego_direction == 'straight' and other_direction == 'left') or (ego_direction == 'left' and other_direction == 'straight')):
                    scenario_type = subtype_name + ' ' + 'straight left side'
                elif spawn_side == 'opposite' and ((ego_direction == 'straight' and other_direction == 'left') or (ego_direction == 'left' and other_direction == 'straight')):
                    scenario_type = subtype_name + ' ' + 'straight left opposite'
                elif spawn_side == 'opposite' and ((ego_direction == 'left' and other_direction == 'right') or (ego_direction == 'right' and other_direction == 'left')):
                    scenario_type = subtype_name + ' ' + 'left right opposite'
            else:
                if ego_direction == 'straight' or other_direction == 'straight':
                    scenario_type = subtype_name + ' ' + 'three-way straight'
                else:
                    scenario_type = subtype_name + ' ' + 'three-way both turns'
        scenario_type_list.append(scenario_type)


scenario_length_array = np.array(scenario_length_list)

print(np.histogram(scenario_length_array, 7, (30, 100)))

pdb.set_trace()
scenario_type_list
scenario_type_dict = {}
for scenario_type_single in scenario_type_list:
    if scenario_type_single not in scenario_type_dict.keys():
        scenario_type_dict[scenario_type_single] = 1
    else:
        scenario_type_dict[scenario_type_single] += 1
pdb.set_trace()


weather_name_array = np.array(weather_name_list)
crash_array = np.array(crash_list)

# weather_list = ['Clear', 'Cloudy', 'HardRain', 'MidRain', 'SoftRain', 'Wet', 'WetCloudy']
# weather_list = ['Clear', 'Cloudy', 'HardRain', 'MidRain', 'SoftRain', 'Wet', 'WetCloudy']
# time_of_day_list = ['Noon', 'Sunset', 'Night']

time_dict_stat = {}
for weather_and_time_name in weather_name_array:
    time = re.findall('[A-Z][^A-Z]*', weather_and_time_name)[-1]
    if time not in time_dict_stat.keys():
        time_dict_stat[time] = 1
    else:
        time_dict_stat[time] += 1
for key in time_dict_stat.keys():
    # weather_dict_stat[key] /= len(weather_name_array)
    print(key, time_dict_stat[key])

weather_dict_stat = {}
for weather_and_time_name in weather_name_array:
    weather = ''.join(re.findall('[A-Z][^A-Z]*', weather_and_time_name)[:-1])
    if weather not in weather_dict_stat.keys():
        weather_dict_stat[weather] = 1
    else:
        weather_dict_stat[weather] += 1

for key in weather_dict_stat.keys():
    # weather_dict_stat[key] /= len(weather_name_array)
    print(key, weather_dict_stat[key])
pdb.set_trace()

time_dict_stat_frame = {}
for idx, weather_and_time_name in enumerate(weather_name_array):
    time = re.findall('[A-Z][^A-Z]*', weather_and_time_name)[-1]
    if time not in time_dict_stat_frame.keys():
        time_dict_stat_frame[time] = scenario_length_array[idx]
    else:
        time_dict_stat_frame[time] += scenario_length_array[idx]

for key in time_dict_stat_frame.keys():
    # weather_dict_stat[key] /= len(weather_name_array)
    print(key, time_dict_stat_frame[key])

weather_dict_stat_frame = {}
for idx, weather_and_time_name in enumerate(weather_name_array):
    weather = ''.join(re.findall('[A-Z][^A-Z]*', weather_and_time_name)[:-1])
    if weather not in weather_dict_stat_frame.keys():
        weather_dict_stat_frame[weather] = scenario_length_array[idx]
    else:
        weather_dict_stat_frame[weather] += scenario_length_array[idx]

for key in weather_dict_stat_frame.keys():
    # weather_dict_stat[key] /= len(weather_name_array)
    print(key, weather_dict_stat_frame[key])
pdb.set_trace()

print('crash percentage: %f'%(np.sum(crash_array)/len(crash_array)))
pdb.set_trace()

import matplotlib.pyplot as plt

# Pie chart
labels = ['Frogs', 'Hogs', 'Dogs', 'Logs']
sizes = [15, 30, 45, 10]
# colors
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
# explsion
explode = (0.05, 0.05, 0.05, 0.05)

plt.pie(sizes, colors=colors, labels=labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85, explode=explode)
# draw circle
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle
# ax1.axis('equal')
plt.tight_layout()
plt.show()

pdb.set_trace()

    # self_bbox_list = []
    # for line in lines:
    #     if len(line.split(' ')) <= 1:
    #         continue
    #     cls_label = line.split(' ')[0]
    #     label = line.split(' ')[1:8]
    #     label_float = list(map(float, label))
    #     # label_float[:3] = np.dot(np.array(label_float[:3] + [1]), transform_matrix)[:3]
    #     label_float[:3] = np.dot(transform_matrix, np.array(label_float[:3] + [1]))[:3]
    #     label_float[6] += yaw
    #
    #     # label_float[6] = -label_float[6]
    #
    #     self_bbox_list.append((cls_label, label_float))