import os
import pdb
import glob
import random
# import numpy.random as random
from collections import Counter
import argparse

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def exluded_list(total_list, common_list):
    list_return = [value for value in total_list if value not in common_list]
    return list_return

def add_type_name_to_scenario(scenario_list, type_name):
    scenario_with_type = []
    for scenario_single in scenario_list:
        scenario_with_type.append((type_name, scenario_single))
    return scenario_with_type


argparser = argparse.ArgumentParser(
        description='DeepAccident Split data')
argparser.add_argument(
    '--data_split',
    default='train', type=str)
argparser.add_argument(
    '--seed',
    default=17, type=int)
argparser.add_argument(
    '--train_ratio',
    default=0.2, type=float)
argparser.add_argument(
    '--train_ratio_tp',
    default=0.4, type=float)

args = argparser.parse_args()

train_ratio = args.train_ratio
train_ratio_tp = args.train_ratio_tp
data_split = args.data_split

# val_ratio_tp = args.val_ratio_tp
# test_ratio_tp = args.val_ratio_tp

random.seed(args.seed)

with open('./data/DeepAccident_data/' + data_split + '.txt', 'r') as f:
    train_list_full = [(line.rstrip().split(' ')[0], line.rstrip().split(' ')[1]) for line in f]

train_list = []
common_list_between_normal_accident = []
file_list_type1_normal = []
file_list_type1_accident = []
file_list_type2_normal = []
file_list_type2_accident = []
exclude_list_type1_normal = []
exclude_list_type1_accident = []
exclude_list_type2_normal = []
exclude_list_type2_accident = []

for train_scenario_single in train_list_full:
    if train_scenario_single[0] == 'type1_subtype1_normal':
        file_list_type1_normal.append(train_scenario_single[1])
    elif train_scenario_single[0] == 'type1_subtype1_accident':
        file_list_type1_accident.append(train_scenario_single[1])
    elif train_scenario_single[0] == 'type1_subtype2_normal':
        file_list_type2_normal.append(train_scenario_single[1])
    elif train_scenario_single[0] == 'type1_subtype2_accident':
        file_list_type2_accident.append(train_scenario_single[1])

common_list_type1 = intersection(file_list_type1_normal, file_list_type1_accident)
exclude_list_type1_normal = exluded_list(file_list_type1_normal, common_list_type1)
exclude_list_type1_accident = exluded_list(file_list_type1_accident, common_list_type1)
exclude_list_type1_normal = add_type_name_to_scenario(exclude_list_type1_normal, 'type1_subtype1_normal')
exclude_list_type1_accident = add_type_name_to_scenario(exclude_list_type1_accident, 'type1_subtype1_accident')
for scenario_name in common_list_type1:
    common_list_between_normal_accident.append(
        (('type1_subtype1_normal', scenario_name),
         ('type1_subtype1_accident', scenario_name)))

common_list_type2 = intersection(file_list_type2_normal, file_list_type2_accident)
exclude_list_type2_normal = exluded_list(file_list_type2_normal, common_list_type2)
exclude_list_type2_accident = exluded_list(file_list_type2_accident, common_list_type2)
exclude_list_type2_normal = add_type_name_to_scenario(exclude_list_type2_normal, 'type1_subtype2_normal')
exclude_list_type2_accident = add_type_name_to_scenario(exclude_list_type2_accident, 'type1_subtype2_accident')
for scenario_name in common_list_type2:
    common_list_between_normal_accident.append(
        (('type1_subtype2_normal', scenario_name),
         ('type1_subtype2_accident', scenario_name)))

print(len(common_list_between_normal_accident), len(exclude_list_type1_normal), len(exclude_list_type1_accident),
      len(exclude_list_type2_normal), len(exclude_list_type2_accident))
print(len(common_list_between_normal_accident))

total_sequence = len(file_list_type1_normal) + len(file_list_type1_accident) \
                 + len(file_list_type2_normal) + len(file_list_type2_accident)

# random.shuffle(common_list_between_normal_accident)
train_len = round(train_ratio*total_sequence)
print('total: %d mini: %d'%(total_sequence, train_len))
# pdb.set_trace()

tp_common_list = []
for elem in common_list_between_normal_accident:
    tp_flag = True
    for scenario_elem in elem:
        scenario_type = scenario_elem[0]
        meta_file = './data/DeepAccident_data/' + scenario_type + '/meta/' + scenario_elem[1] + '.txt'
        with open(meta_file) as f:
            lines = [line.rstrip('\n') for line in f]
        collision_status = lines[0].split(' ')[1:8]
        colliding_agents = lines[1].split('colliding agents: ')[1]
        colliding_agent1 = colliding_agents.split(' ')[0]
        colliding_agent2 = colliding_agents.split(' ')[1]

        collision_agents_id = ('ego' in colliding_agent1 and 'other' in colliding_agent2) or \
                    ('other' in colliding_agent1 and 'ego' in colliding_agent2)
        if 'accident' in scenario_type:
            if collision_status != ['-1', '-1', '-1', '-1', '-1', '-1', '-1']:
                if collision_agents_id:
                    pass
                else:
                    tp_flag = False
                    break
                pass
            else:
                tp_flag = False
                break
        else:
            if collision_status == ['-1', '-1', '-1', '-1', '-1', '-1', '-1']:
                pass
            else:
                tp_flag = False
                break

    if tp_flag:
        tp_common_list.append(elem)
        common_list_between_normal_accident.remove(elem)
# print(len(tp_common_list), len(common_list_between_normal_accident))
# pdb.set_trace()

def category_tp_dict(tp_common_list):
    tp_common_list_three_way = []
    tp_common_list_four_way_opposite = []
    tp_common_list_four_way_not_opposite = []

    for elem in tp_common_list:
        tp_dict = {}
        scenario_elem = elem[1]
        scenario_type = scenario_elem[0]
        meta_file = './data/DeepAccident_data/' + scenario_type + '/meta/' + scenario_elem[1] + '.txt'
        with open(meta_file) as f:
            lines = [line.rstrip('\n') for line in f]

        road_type = lines[3].split(': ')[1].split(' junction')[0]
        spawn_side = lines[4].split(': ')[1]
        if road_type == 'three-way':
            tp_common_list_three_way.append(elem)
        else:
            if spawn_side == 'opposite':
                tp_common_list_four_way_opposite.append(elem)
            else:
                tp_common_list_four_way_not_opposite.append(elem)

    return tp_common_list_three_way, tp_common_list_four_way_opposite, tp_common_list_four_way_not_opposite

def analyze_data_split(data_list):
    tp_dict_list = []
    for scenario_elem in data_list:
        tp_dict = {}
        scenario_type = scenario_elem[0]
        meta_file = './data/DeepAccident_data/' + scenario_type + '/meta/' + scenario_elem[1] + '.txt'
        with open(meta_file) as f:
            lines = [line.rstrip('\n') for line in f]
        tp_dict['scenario_name'] = scenario_elem[1]
        if 'Noon' in lines[0].split(' ')[0]:
            tp_dict['time_of_day'] = 'Noon'
        elif 'Night' in lines[0].split(' ')[0]:
            tp_dict['time_of_day'] = 'Night'
        else:
            tp_dict['time_of_day'] = 'Sunset'

        tp_dict['weather'] = lines[0].split(' ')[0].split(tp_dict['time_of_day'])[0]
        tp_dict['scenario_length'] = lines[0].split(' ')[-1]
        tp_dict['town'], _, tp_dict['type'], tp_dict['scenario'] = scenario_elem[1].split('_')

        # colliding_agents = lines[1].split('colliding agents: ')[1]
        # colliding_agent1 = colliding_agents.split(' ')[0]
        # colliding_agent2 = colliding_agents.split(' ')[1]
        # tp_dict['colliding_agent1'] = colliding_agent1
        # tp_dict['colliding_agent2'] = colliding_agent2

        collision_status = lines[0].split(' ')[1:8]
        colliding_agents = lines[1].split('colliding agents: ')[1]
        colliding_agent1 = colliding_agents.split(' ')[0]
        colliding_agent2 = colliding_agents.split(' ')[1]

        collision_agents_id = ('ego' in colliding_agent1 and 'other' in colliding_agent2) or \
                              ('other' in colliding_agent1 and 'ego' in colliding_agent2)

        tp_dict['correct_collision_status'] = False
        tp_dict['correct_collision_agents'] = False
        if 'accident' in scenario_type:
            tp_dict['accident_scenario'] = True
            if collision_status != ['-1', '-1', '-1', '-1', '-1', '-1', '-1']:
                tp_dict['correct_collision_status'] = True
                if collision_agents_id:
                    tp_dict['correct_collision_agents'] = True
        else:
            tp_dict['accident_scenario'] = False
            if collision_status == ['-1', '-1', '-1', '-1', '-1', '-1', '-1']:
                tp_dict['correct_collision_status'] = True
                tp_dict['correct_collision_agents'] = True

        road_type = lines[3].split(': ')[1].split(' junction')[0]
        spawn_side = lines[4].split(': ')[1]
        self_direction = lines[5].split(': ')[1]
        other_direction = lines[6].split(': ')[1]
        tp_dict['road_type'] = road_type
        tp_dict['spawn_side'] = 'opposite' if spawn_side == 'opposite' else 'not opposite'
        tp_dict['self_direction'] = self_direction
        tp_dict['other_direction'] = other_direction
        tp_dict_list.append(tp_dict)

    tp_dict_list = list(sorted(tp_dict_list, key=lambda x: (x['type'], x['town'], x['road_type'], x['spawn_side'], x['scenario'])))

    total_list_len = len(tp_dict_list)
    keys_to_analyze = ['town', 'accident_scenario', 'correct_collision_status', 'correct_collision_agents', 'time_of_day', 'weather', 'road_type', 'spawn_side']

    counter_list = []
    for key in keys_to_analyze:
        if key == 'spawn_side':
            tmp_list = []
            for tp_dict_single in tp_dict_list:
                if tp_dict_single['road_type'] == 'four-way':
                    tmp_list.append(tp_dict_single['spawn_side'])
            counter_single = Counter(tmp_list)
        else:
            counter_single = Counter([tp_dict_single[key] for tp_dict_single in tp_dict_list])
        for key_tmp, value_tmp in counter_single.items():
            print(key, key_tmp, value_tmp, value_tmp/total_list_len)
        counter_list.append(counter_single)

    print('\n')
    return tp_dict_list

def append_list_by_ratio_paired(src_paired_list, train_list, train_ratio, train_max):
    # intial_len = len(train_list)
    random.shuffle(src_paired_list)
    len_total = len(src_paired_list)
    remaining_train_len = (train_max - len(train_list)) // 2
    train_len = round(len_total * train_ratio)
    train_len = min(train_len, remaining_train_len)
    for i in range(0, train_len):
        train_list.append(src_paired_list[i][0])
        train_list.append(src_paired_list[i][1])
    # print(len(train_list) - intial_len)

def append_list_by_ratio(src_list, train_list, train_ratio, train_max):
    # intial_len = len(train_list)
    random.shuffle(src_list)
    len_total = len(src_list)
    remaining_train_len = train_max - len(train_list)
    train_len = round(len_total * train_ratio)
    train_len = min(train_len, remaining_train_len)
    for i in range(0, train_len):
        train_list.append(src_list[i])
    # print(len(train_list) - intial_len)

tp_common_list_three_way, tp_common_list_four_way_opposite, tp_common_list_four_way_not_opposite = category_tp_dict(tp_common_list)
append_list_by_ratio_paired(tp_common_list_three_way, train_list, train_ratio_tp, train_len)
append_list_by_ratio_paired(tp_common_list_four_way_opposite, train_list, train_ratio_tp, train_len)
append_list_by_ratio_paired(tp_common_list_four_way_not_opposite, train_list, train_ratio_tp, train_len)

not_tp_common_list_three_way, not_tp_common_list_four_way_opposite, not_tp_common_list_four_way_not_opposite = category_tp_dict(common_list_between_normal_accident)
append_list_by_ratio_paired(not_tp_common_list_three_way, train_list, train_ratio, train_len)
append_list_by_ratio_paired(not_tp_common_list_four_way_opposite, train_list, train_ratio, train_len)
append_list_by_ratio_paired(not_tp_common_list_four_way_not_opposite, train_list, train_ratio, train_len)

append_list_by_ratio(exclude_list_type1_normal, train_list, train_ratio, train_len)
append_list_by_ratio(exclude_list_type1_accident, train_list, train_ratio, train_len)
append_list_by_ratio(exclude_list_type2_normal, train_list, train_ratio, train_len)
append_list_by_ratio(exclude_list_type2_accident, train_list, train_ratio, train_len)

print(len(train_list))
pdb.set_trace()

random.shuffle(train_list)
analyzed_dict_list_val = analyze_data_split(train_list)

# pdb.set_trace()
with open('./data/DeepAccident_data/' + data_split + '_mini.txt', 'w') as f:
    for scenario_elem in train_list:
        # if scenario_elem[0] == "type1_subtype1_accident" and scenario_elem[1] == "Town03_type001_subtype0001_scenario00030":
        #     continue
        # if scenario_elem[0] == "type1_subtype1_accident" and scenario_elem[1] == "Town04_type001_subtype0001_scenario00030":
        #     continue
        f.write("%s %s\n" % (scenario_elem[0], scenario_elem[1]))
