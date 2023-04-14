import os
import pdb
import glob
import random
# import numpy.random as random
from collections import Counter
import argparse

def count_file_number(dir_path):
    count = 0
    for path in os.scandir(dir_path):
        if path.is_file():
            count += 1
    return count

def total_samples_in_split(file_number_dict, split_list):
    count = 0
    for scenario in split_list:
        count += file_number_dict[scenario]
    return count


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def exluded_list(total_list, common_list):
    list_return = [value for value in total_list if value not in common_list]
    return list_return

def append_exluded_list(train_list, val_list, test_list, scenario_type,
                            scenario_name):
    number_rand = random.uniform(0, 1)
    # if number_rand >= 0.4 and number_rand < 0.5:
    #     val_list.append((scenario_type, scenario_name))
    # elif number_rand >= 0.5 and number_rand < 0.6:
    #     test_list.append((scenario_type, scenario_name))
    # else:
    #     train_list.append((scenario_type, scenario_name))

    train_list.append((scenario_type, scenario_name))

def add_type_name_to_scenario(scenario_list, type_name):
    scenario_with_type = []
    for scenario_single in scenario_list:
        scenario_with_type.append((type_name, scenario_single))
    return scenario_with_type


argparser = argparse.ArgumentParser(
        description='DeepAccident Split data')
argparser.add_argument(
    '--seed',
    default=17, type=int)
argparser.add_argument(
    '--val_ratio_tp',
    default=0.4, type=float)

args = argparser.parse_args()

val_ratio_tp = args.val_ratio_tp
test_ratio_tp = args.val_ratio_tp

random.seed(args.seed)

train_list = []
val_list = []
test_list = []
common_list_between_normal_accident = []
file_list_type1_normal = []
file_list_type1_accident = []
file_list_type2_normal = []
file_list_type2_accident = []
file_list_type3_normal = []
file_list_type3_accident = []

exclude_list_type1_normal = []
exclude_list_type1_accident = []
exclude_list_type2_normal = []
exclude_list_type2_accident = []
exclude_list_type3_normal = []
exclude_list_type3_accident = []

scenario_type = 'type1_subtype1_accident'
rootdir = './data/DeepAccident_data/' + scenario_type + '/ego_vehicle/label'
type1_exists = os.path.exists(rootdir)
if type1_exists:
    file_number_dict = {}
    file_list_type1_accident = []
    scenario_list = glob.glob(f'{rootdir}/*/')
    # random.shuffle(scenario_list)
    for path in scenario_list:
        file_number = count_file_number(path)
        # # print(path.split('/')[-2], file_number)

        file_number_dict[path.split('/')[-2]] = file_number
        if file_number > 5:
            file_list_type1_accident.append(path.split('/')[-2])

    scenario_type = 'type1_subtype1_normal'
    rootdir = './data/DeepAccident_data/' + scenario_type + '/ego_vehicle/label'
    file_number_dict = {}
    file_list_type1_normal = []
    scenario_list = glob.glob(f'{rootdir}/*/')
    # random.shuffle(scenario_list)
    for path in scenario_list:
        file_number = count_file_number(path)
        # # print(path.split('/')[-2], file_number)

        file_number_dict[path.split('/')[-2]] = file_number
        if file_number > 5:
            file_list_type1_normal.append(path.split('/')[-2])

    common_list_type1 = intersection(file_list_type1_normal, file_list_type1_accident)
    exclude_list_type1_normal = exluded_list(file_list_type1_normal, common_list_type1)
    exclude_list_type1_accident = exluded_list(file_list_type1_accident, common_list_type1)
    exclude_list_type1_normal = add_type_name_to_scenario(exclude_list_type1_normal, 'type1_subtype1_normal')
    exclude_list_type1_accident = add_type_name_to_scenario(exclude_list_type1_accident, 'type1_subtype1_accident')

    # for scenario_name in exclude_list_type1_normal:
    #     append_exluded_list(train_list, val_list, test_list, 'type1_subtype1_normal',
    #                         scenario_name)
    # for scenario_name in exclude_list_type1_accident:
    #     append_exluded_list(train_list, val_list, test_list, 'type1_subtype1_accident',
    #                         scenario_name)
    for scenario_name in common_list_type1:
        common_list_between_normal_accident.append(
            (('type1_subtype1_normal', scenario_name),
             ('type1_subtype1_accident', scenario_name)))

scenario_type = 'type1_subtype2_accident'
rootdir = './data/DeepAccident_data/' + scenario_type + '/ego_vehicle/label'
type2_exists = os.path.exists(rootdir)
if type2_exists:
    file_number_dict = {}
    file_list_type2_accident = []
    scenario_list = glob.glob(f'{rootdir}/*/')
    # random.shuffle(scenario_list)
    for path in scenario_list:
        file_number = count_file_number(path)
        # print(path.split('/')[-2], file_number)

        file_number_dict[path.split('/')[-2]] = file_number
        if file_number > 5:
            # if path.split('/')[-2].startswith('Town04') and int(path.split('/')[-2].split('scenario')[-1]) >= 27:
            #     print(path.split('/')[-2])
            #     continue
            file_list_type2_accident.append(path.split('/')[-2])

    scenario_type = 'type1_subtype2_normal'
    rootdir = './data/DeepAccident_data/' + scenario_type + '/ego_vehicle/label'
    file_number_dict = {}
    file_list_type2_normal = []
    scenario_list = glob.glob(f'{rootdir}/*/')
    # random.shuffle(scenario_list)
    for path in scenario_list:
        file_number = count_file_number(path)
        # print(path.split('/')[-2], file_number)

        file_number_dict[path.split('/')[-2]] = file_number
        if file_number > 5:
            file_list_type2_normal.append(path.split('/')[-2])

    common_list_type2 = intersection(file_list_type2_normal, file_list_type2_accident)
    exclude_list_type2_normal = exluded_list(file_list_type2_normal, common_list_type2)
    exclude_list_type2_accident = exluded_list(file_list_type2_accident, common_list_type2)
    exclude_list_type2_normal = add_type_name_to_scenario(exclude_list_type2_normal, 'type1_subtype2_normal')
    exclude_list_type2_accident = add_type_name_to_scenario(exclude_list_type2_accident, 'type1_subtype2_accident')
    # for scenario_name in exclude_list_type2_normal:
    #     append_exluded_list(train_list, val_list, test_list, 'type1_subtype2_normal',
    #                         scenario_name)
    # for scenario_name in exclude_list_type2_accident:
    #     append_exluded_list(train_list, val_list, test_list, 'type1_subtype2_accident',
    #                         scenario_name)
    for scenario_name in common_list_type2:
        common_list_between_normal_accident.append(
            (('type1_subtype2_normal', scenario_name),
             ('type1_subtype2_accident', scenario_name)))

scenario_type = 'type2_subtype1_accident'
rootdir = './data/DeepAccident_data/' + scenario_type + '/ego_vehicle/label'
type3_exists = os.path.exists(rootdir)
if type3_exists:
    file_number_dict = {}
    file_list_type3_accident = []
    scenario_list = glob.glob(f'{rootdir}/*/')
    # random.shuffle(scenario_list)
    for path in scenario_list:
        file_number = count_file_number(path)
        # print(path.split('/')[-2], file_number)

        file_number_dict[path.split('/')[-2]] = file_number
        if file_number > 5:
            file_list_type3_accident.append(path.split('/')[-2])

    scenario_type = 'type2_subtype1_normal'
    rootdir = './data/DeepAccident_data/' + scenario_type + '/ego_vehicle/label'
    file_number_dict = {}
    file_list_type3_normal = []
    scenario_list = glob.glob(f'{rootdir}/*/')
    # random.shuffle(scenario_list)
    for path in scenario_list:
        file_number = count_file_number(path)
        # print(path.split('/')[-2], file_number)

        file_number_dict[path.split('/')[-2]] = file_number
        if file_number > 5:
            file_list_type3_normal.append(path.split('/')[-2])

    common_list_type3 = intersection(file_list_type3_normal, file_list_type3_accident)
    exclude_list_type3_normal = exluded_list(file_list_type3_normal, common_list_type3)
    exclude_list_type3_accident = exluded_list(file_list_type3_accident, common_list_type3)
    exclude_list_type3_normal = add_type_name_to_scenario(exclude_list_type3_normal, 'type2_subtype1_normal')
    exclude_list_type3_accident = add_type_name_to_scenario(exclude_list_type3_accident, 'type2_subtype1_accident')

    # for scenario_name in exclude_list_type3_normal:
    #     append_exluded_list(train_list, val_list, test_list, 'type2_subtype1_normal',
    #                         scenario_name)
    # for scenario_name in exclude_list_type3_accident:
    #     append_exluded_list(train_list, val_list, test_list, 'type2_subtype1_accident',
    #                         scenario_name)
    for scenario_name in common_list_type3:
        common_list_between_normal_accident.append(
            (('type2_subtype1_normal', scenario_name),
             ('type2_subtype1_accident', scenario_name)))

total_sequence = len(file_list_type1_normal) + len(file_list_type1_accident) \
                 + len(file_list_type2_normal) + len(file_list_type2_accident) \
                 + len(file_list_type3_normal) + len(file_list_type3_accident)

# random.shuffle(common_list_between_normal_accident)
val_ratio = 0.15
test_ratio = 0.15
val_len = round(val_ratio*total_sequence)
test_len = round(test_ratio*total_sequence)
train_len = total_sequence - val_len - test_len

print(len(common_list_between_normal_accident))
tp_common_list = []
for elem in common_list_between_normal_accident:
    # scenario_normal = elem[0]
    # scenario_accident = elem[1]
    tp_flag = True
    for scenario_elem in elem:
        scenario_type = scenario_elem[0]
        meta_file = './data/DeepAccident_data/' + scenario_type + '/meta/' + scenario_elem[1] + '.txt'
        with open(meta_file) as f:
            lines = [line.rstrip('\n') for line in f]
        # print(lines[0])
        # if 'pedestrian' in lines[0].split(' '):
        #     desired += 1
        #     print(meta_file)
        collision_status = lines[0].split(' ')[1:8]
        colliding_agents = lines[1].split('colliding agents: ')[1]
        colliding_agent1 = colliding_agents.split(' ')[0]
        colliding_agent2 = colliding_agents.split(' ')[1]

        collision_agents_id = ('ego' in colliding_agent1 and 'other' in colliding_agent2) or \
                    ('other' in colliding_agent1 and 'ego' in colliding_agent2)
        if 'accident' in scenario_type:
            if collision_status != ['-1', '-1', '-1', '-1', '-1', '-1', '-1']:
                if collision_agents_id:
                    print(scenario_elem)
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

def category_tp_dict(tp_common_list):
    tp_common_list_three_way = []
    tp_common_list_four_way_opposite = []
    tp_common_list_four_way_not_opposite = []

    for elem in tp_common_list:
        tp_dict = {}
        # print(elem[1][1])
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
        # # self_direction = lines[5].split(': ')[1]
        # # other_direction = lines[6].split(': ')[1]
        # tp_dict['road_type'] = road_type
        # tp_dict['spawn_side'] = 'opposite' if spawn_side == 'opposite' else 'not opposite'
        # tp_dict['self_direction'] = self_direction
        # tp_dict['other_direction'] = other_direction
        # tp_dict_list.append(tp_dict)

    # tp_dict_list = list(
    #     sorted(tp_dict_list, key=lambda x: (x['type'], x['town'], x['road_type'], x['spawn_side'], x['scenario'])))
    return tp_common_list_three_way, tp_common_list_four_way_opposite, tp_common_list_four_way_not_opposite

def get_tp_dict(tp_common_list):
    tp_dict_list = []
    for elem in tp_common_list:
        tp_dict = {}
        # print(elem[1][1])
        scenario_elem = elem[1]
        scenario_type = scenario_elem[0]
        meta_file = './data/DeepAccident_data/' + scenario_type + '/meta/' + scenario_elem[1] + '.txt'
        with open(meta_file) as f:
            lines = [line.rstrip('\n') for line in f]
        tp_dict['scenario_name'] = scenario_elem[1]
        tp_dict['weather'] = lines[0].split(' ')[0]
        tp_dict['scenario_length'] = lines[0].split(' ')[-1]
        tp_dict['town'], _, tp_dict['type'], tp_dict['scenario'] = elem[1][1].split('_')

        colliding_agents = lines[1].split('colliding agents: ')[1]
        colliding_agent1 = colliding_agents.split(' ')[0]
        colliding_agent2 = colliding_agents.split(' ')[1]
        tp_dict['colliding_agent1'] = colliding_agent1
        tp_dict['colliding_agent2'] = colliding_agent2

        road_type = lines[3].split(': ')[1].split(' junction')[0]
        spawn_side = lines[4].split(': ')[1]
        self_direction = lines[5].split(': ')[1]
        other_direction = lines[6].split(': ')[1]
        tp_dict['road_type'] = road_type
        tp_dict['spawn_side'] = 'opposite' if spawn_side == 'opposite' else 'not opposite'
        tp_dict['self_direction'] = self_direction
        tp_dict['other_direction'] = other_direction
        tp_dict_list.append(tp_dict)
        # print(road_type, spawn_side, self_direction, other_direction)
        # print('\n')
    tp_dict_list = list(sorted(tp_dict_list, key=lambda x: (x['type'], x['town'], x['road_type'], x['spawn_side'], x['scenario'])))
    return tp_dict_list

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
        # print(road_type, spawn_side, self_direction, other_direction)
        # print('\n')
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

def append_list_by_ratio_paired(src_paired_list, train_list, val_list, test_list, val_ratio, test_ratio, val_max, test_max,
                                val_len=None, test_len=None):
    random.shuffle(src_paired_list)
    len_total = len(src_paired_list)
    remaining_val_len = (val_max - len(val_list)) // 2
    remaining_test_len = (test_max - len(test_list)) // 2
    if not val_len:
        val_len = round(len_total * val_ratio)
    if not test_len:
        test_len = round(len_total * test_ratio)
    val_len = min(val_len, remaining_val_len)
    test_len = min(test_len, remaining_test_len)
    for i in range(0, val_len):
        val_list.append(src_paired_list[i][0])
        val_list.append(src_paired_list[i][1])
    for i in range(val_len, val_len+test_len):
        test_list.append(src_paired_list[i][0])
        test_list.append(src_paired_list[i][1])
    for i in range(val_len+test_len, len_total):
        train_list.append(src_paired_list[i][0])
        train_list.append(src_paired_list[i][1])

def append_list_by_ratio(src_list, train_list, val_list, test_list, val_ratio, test_ratio, val_max,
                         test_max, val_len=None, test_len=None):
    random.shuffle(src_list)
    len_total = len(src_list)
    remaining_val_len = val_max - len(val_list)
    remaining_test_len = test_max - len(test_list)
    if not val_len:
        val_len = round(len_total * val_ratio)
    if not test_len:
        test_len = round(len_total * test_ratio)
    val_len = min(val_len, remaining_val_len)
    test_len = min(test_len, remaining_test_len)
    for i in range(0, val_len):
        val_list.append(src_list[i])
    for i in range(val_len, val_len+test_len):
        test_list.append(src_list[i])
    for i in range(val_len+test_len, len_total):
        train_list.append(src_list[i])


# tp_dict_list = get_tp_dict(tp_common_list)
# print('\n')
# for aa in tp_dict_list:
#     print(list(aa.values()))

print(len(val_list), len(test_list), len(train_list))
current_total = len(val_list) + len(test_list) + len(train_list)
if current_total:
    print(current_total)
    print(len(val_list)/current_total, len(test_list)/current_total, len(train_list)/current_total)
else:
    print(current_total)
    print(0, 0, 0)
# pdb.set_trace()

# val_ratio_tp = 0.35
# test_ratio_tp = 0.35
tp_common_list_three_way, tp_common_list_four_way_opposite, tp_common_list_four_way_not_opposite = category_tp_dict(tp_common_list)
append_list_by_ratio_paired(tp_common_list_three_way, train_list, val_list, test_list, val_ratio_tp, test_ratio_tp, val_len, test_len)
append_list_by_ratio_paired(tp_common_list_four_way_opposite, train_list, val_list, test_list, val_ratio_tp, test_ratio_tp, val_len, test_len)
append_list_by_ratio_paired(tp_common_list_four_way_not_opposite, train_list, val_list, test_list, val_ratio_tp, test_ratio_tp, val_len, test_len)

print(len(val_list), len(test_list), len(train_list))
current_total = len(val_list) + len(test_list) + len(train_list)
if current_total:
    print(current_total)
    print(len(val_list)/current_total, len(test_list)/current_total, len(train_list)/current_total)
else:
    print(current_total)
    print(0, 0, 0)
# pdb.set_trace()

not_tp_common_list_three_way, not_tp_common_list_four_way_opposite, not_tp_common_list_four_way_not_opposite = category_tp_dict(common_list_between_normal_accident)
append_list_by_ratio_paired(not_tp_common_list_three_way, train_list, val_list, test_list, val_ratio, test_ratio, val_len, test_len)
append_list_by_ratio_paired(not_tp_common_list_four_way_opposite, train_list, val_list, test_list, val_ratio, test_ratio, val_len, test_len)
append_list_by_ratio_paired(not_tp_common_list_four_way_not_opposite, train_list, val_list, test_list, val_ratio, test_ratio, val_len, test_len)

# pdb.set_trace()
# print((('type1_subtype2_normal', 'Town04_type001_subtype0002_scenario00034'), ('type1_subtype2_accident', 'Town04_type001_subtype0002_scenario00034')) in not_tp_common_list_three_way)

# append_list_by_ratio_paired(common_list_between_normal_accident, train_list, val_list, test_list, val_ratio, test_ratio)

# tp_dict_list = get_tp_dict(common_list_between_normal_accident)
# print('\n')
# for aa in tp_dict_list:
#     print(list(aa.values()))

print(len(val_list), len(test_list), len(train_list))
current_total = len(val_list) + len(test_list) + len(train_list)
if current_total:
    print(current_total)
    print(len(val_list)/current_total, len(test_list)/current_total, len(train_list)/current_total)
else:
    print(current_total)
    print(0, 0, 0)
# pdb.set_trace()

print(len(common_list_between_normal_accident), len(tp_common_list))

#  another_vehicle_spawn_side: opposite
#  ego_vehicle_direction: straight

# # type 1, four-way, spawn left/right, direction straight & straight
# specical_case1 = False
# # type 2, four-way, spawn left/right, direction straight & straight
# specical_case2 = False
# # type 2, highway merge
# specical_case3 = False
#
# for elem in tp_common_list:
#     scenario_elem = elem[1]
#     scenario_type = scenario_elem[0]
#     meta_file = './data/DeepAccident_data/' + scenario_type + '/meta/' + scenario_elem[1] + '.txt'
#     town_name = scenario_elem[1].split('_')[0]
#
#     with open(meta_file) as f:
#         lines = [line.rstrip('\n') for line in f]
#     collision_status = lines[0].split(' ')[1:8]
#     colliding_agents = lines[1].split('colliding agents: ')[1]
#     road_type = lines[3].split(': ')[1].split(' junction')[0]
#     spawn_side = lines[4].split(': ')[1]
#     self_direction = lines[5].split(': ')[1]
#     other_direction = lines[6].split(': ')[1]
#     if not specical_case1 and 'type1_subtype1' in scenario_type and road_type == 'four-way' and spawn_side != 'opposite' \
#             and self_direction == 'straight' and other_direction == 'straight':
#         val_list.append(elem[0])
#         val_list.append(elem[1])
#         tp_common_list.remove(elem)
#         specical_case1 = True
#
#     elif not specical_case2 and 'type1_subtype2' in scenario_type and road_type == 'four-way' and spawn_side != 'opposite' \
#             and self_direction == 'straight' and other_direction == 'straight':
#         val_list.append(elem[0])
#         val_list.append(elem[1])
#         tp_common_list.remove(elem)
#         specical_case2 = True
#
#     elif not specical_case3 and 'type1_subtype2' in scenario_type and scenario_elem[1] == 'Town04_type001_subtype0002_scenario00034':
#         val_list.append(elem[0])
#         val_list.append(elem[1])
#         tp_common_list.remove(elem)
#         specical_case3 = True
#
#
#
#     if specical_case1 and specical_case2 and specical_case3:
#         break
#
#
# # type 1, four-way, spawn left/right, direction straight & straight
# specical_case1 = False
# # type 2, four-way, spawn left/right, direction straight & straight
# specical_case2 = False
# # type 2, highway merge
# specical_case3 = False
# for elem in tp_common_list:
#     scenario_elem = elem[1]
#     scenario_type = scenario_elem[0]
#     meta_file = './data/DeepAccident_data/' + scenario_type + '/meta/' + scenario_elem[1] + '.txt'
#     town_name = scenario_elem[1].split('_')[0]
#     with open(meta_file) as f:
#         lines = [line.rstrip('\n') for line in f]
#     collision_status = lines[0].split(' ')[1:8]
#     colliding_agents = lines[1].split('colliding agents: ')[1]
#     road_type = lines[3].split(': ')[1].split(' junction')[0]
#     spawn_side = lines[4].split(': ')[1]
#     self_direction = lines[5].split(': ')[1]
#     other_direction = lines[6].split(': ')[1]
#     if not specical_case1 and 'type1_subtype1' in scenario_type and road_type == 'four-way' and spawn_side != 'opposite' \
#             and self_direction == 'straight' and other_direction == 'straight':
#         test_list.append(elem[0])
#         test_list.append(elem[1])
#         tp_common_list.remove(elem)
#         specical_case1 = True
#
#     elif not specical_case2 and 'type1_subtype2' in scenario_type and road_type == 'four-way' and spawn_side != 'opposite' \
#             and self_direction == 'straight' and other_direction == 'straight':
#         test_list.append(elem[0])
#         test_list.append(elem[1])
#         tp_common_list.remove(elem)
#         specical_case2 = True
#
#     elif not specical_case3 and 'type1_subtype2' in scenario_type and scenario_elem[1] == 'Town04_type001_subtype0002_scenario00039':
#         test_list.append(elem[0])
#         test_list.append(elem[1])
#         tp_common_list.remove(elem)
#         specical_case3 = True
#
#     if specical_case1 and specical_case2 and specical_case3:
#         break


# # remaining_train_len = train_len - len(train_list)
# remaining_val_len = val_len - len(val_list)
# remaining_test_len = test_len - len(test_list)
# print(remaining_val_len, remaining_test_len)
#
# # train_len_half = remaining_train_len // 2
# val_len_half = remaining_val_len // 2
# test_len_half = remaining_test_len // 2
#
# if len(tp_common_list) > val_len_half + test_len_half:
#     for elem in tp_common_list[:val_len_half]:
#         val_list.append(elem[0])
#         val_list.append(elem[1])
#
#     for elem in tp_common_list[val_len_half:val_len_half+test_len_half]:
#         test_list.append(elem[0])
#         test_list.append(elem[1])
#
#     for elem in tp_common_list[val_len_half+test_len_half:]:
#         train_list.append(elem[0])
#         train_list.append(elem[1])
# else:
#     one_third_tp = len(tp_common_list) // 3
#     for elem in tp_common_list[:one_third_tp]:
#         val_list.append(elem[0])
#         val_list.append(elem[1])
#
#     for elem in tp_common_list[one_third_tp:2*one_third_tp]:
#         test_list.append(elem[0])
#         test_list.append(elem[1])
#
#     for elem in tp_common_list[2*one_third_tp:]:
#         train_list.append(elem[0])
#         train_list.append(elem[1])



# # remaining_train_len = train_len - len(train_list)
# remaining_val_len = val_len - len(val_list)
# remaining_test_len = test_len - len(test_list)
# print(remaining_val_len, remaining_test_len)
#
# # train_len_half = remaining_train_len // 2
# val_len_half = remaining_val_len // 2
# test_len_half = remaining_test_len // 2
#
# for elem in common_list_between_normal_accident[:val_len_half]:
#     val_list.append(elem[0])
#     val_list.append(elem[1])
#
# for elem in common_list_between_normal_accident[val_len_half:val_len_half+test_len_half]:
#     test_list.append(elem[0])
#     test_list.append(elem[1])
#
# for elem in common_list_between_normal_accident[val_len_half+test_len_half:]:
#     train_list.append(elem[0])
#     train_list.append(elem[1])


# for scenario_name in exclude_list_type1_normal:
#     append_exluded_list(train_list, val_list, test_list, 'type1_subtype1_normal',
#                         scenario_name)
# for scenario_name in exclude_list_type1_accident:
#     append_exluded_list(train_list, val_list, test_list, 'type1_subtype1_accident',
#                         scenario_name)
#
# for scenario_name in exclude_list_type2_normal:
#     append_exluded_list(train_list, val_list, test_list, 'type1_subtype2_normal',
#                         scenario_name)
# for scenario_name in exclude_list_type2_accident:
#     append_exluded_list(train_list, val_list, test_list, 'type1_subtype2_accident',
#                         scenario_name)
#
# for scenario_name in exclude_list_type3_normal:
#     append_exluded_list(train_list, val_list, test_list, 'type2_subtype1_normal',
#                         scenario_name)
# for scenario_name in exclude_list_type3_accident:
#     append_exluded_list(train_list, val_list, test_list, 'type2_subtype1_accident',
#                         scenario_name)


append_list_by_ratio(exclude_list_type1_normal, train_list, val_list, test_list, val_ratio, test_ratio, val_len, test_len)
append_list_by_ratio(exclude_list_type1_accident, train_list, val_list, test_list, val_ratio, test_ratio, val_len, test_len)


append_list_by_ratio(exclude_list_type2_normal, train_list, val_list, test_list, val_ratio, test_ratio, val_len, test_len)
append_list_by_ratio(exclude_list_type2_accident, train_list, val_list, test_list, val_ratio, test_ratio, val_len, test_len)

# append_list_by_ratio(exclude_list_type3_normal, train_list, val_list, test_list, val_ratio, test_ratio)
# append_list_by_ratio(exclude_list_type3_accident, train_list, val_list, test_list, val_ratio, test_ratio)

print(len(val_list), len(test_list), len(train_list))
current_total = len(val_list) + len(test_list) + len(train_list)
if current_total:
    print(current_total)
    print(len(val_list)/current_total, len(test_list)/current_total, len(train_list)/current_total)
else:
    print(current_total)
    print(0, 0, 0)
# pdb.set_trace()

# four_way_num = 0
# three_way_num = 0
# for scenario_elem in val_list[0::2]:
#     scenario_type = scenario_elem[0]
#     meta_file = './data/DeepAccident_data/' + scenario_type + '/meta/' + scenario_elem[1] + '.txt'
#     print(scenario_type, scenario_elem[1])
#     with open(meta_file) as f:
#         lines = [line.rstrip('\n') for line in f]
#     print(lines[0])
#     print(lines[1])
#     road_type = lines[3].split('road_type: ')[1].split(' junction')[0]
#     if road_type == 'four-way':
#         four_way_num += 1
#     else:
#         three_way_num += 1
#     print(road_type)
#     print(lines[4])
#     print('\n')
# print(four_way_num, three_way_num)
# pdb.set_trace()


if len(intersection(train_list, val_list)) > 0 or len(intersection(val_list, test_list)) > 0 \
        or len(intersection(train_list, val_list)) > 0:
    print('Error: train val test splits have overlaps')

# def count_desired(train_list, debug=False):
#     desired1 = 0
#     desired2 = 0
#     for scenario_elem in train_list:
#         scenario_type = scenario_elem[0]
#         meta_file = './data/DeepAccident_data/' + scenario_type + '/meta/' + scenario_elem[1] + '.txt'
#         with open(meta_file) as f:
#             lines = [line.rstrip('\n') for line in f]
#         # print(lines[0])
#         # if 'pedestrian' in lines[0].split(' '):
#         #     desired += 1
#         #     print(meta_file)
#         collision_status = lines[0].split(' ')[1:8]
#         colliding_agents = lines[1].split('colliding agents: ')[1]
#         colliding_agent1 = colliding_agents.split(' ')[0]
#         colliding_agent2 = colliding_agents.split(' ')[1]
#         if 'accident' in scenario_type:
#             # if collision_status != ['-1', '-1', '-1', '-1', '-1', '-1', '-1']:
#             #     collision_agents_id = False
#             #     if ('ego' in colliding_agent1 and 'other' in colliding_agent2) or \
#             #         ('other' in colliding_agent1 and 'ego' in colliding_agent2):
#             #         collision_agents_id = True
#             #     if collision_agents_id:
#             #         if debug:
#             #             print(lines[0])
#             #         desired1 += 1
#             if collision_status != ['-1', '-1', '-1', '-1', '-1', '-1', '-1']:
#                 if debug:
#                     print(lines[0])
#                 desired1 += 1
#
#         else:
#             if collision_status == ['-1', '-1', '-1', '-1', '-1', '-1', '-1']:
#                 desired2 += 1
#     desired = desired1 + desired2
#     print('\n')
#     return desired, desired/len(train_list), desired1, desired2
#
#
# desired_train, desired_train_ratio, desired_tp_train, desired_tn_train = count_desired(train_list, debug=False)
# desired_val, desired_val_ratio, desired_tp_val, desired_tn_val = count_desired(val_list, debug=True)
# desired_test, desired_test_ratio, desired_tp_test, desired_tn_test = count_desired(test_list, debug=True)
#
# print(desired_train, desired_train_ratio, desired_tp_train, desired_tn_train)
# print(desired_val, desired_val_ratio, desired_tp_val, desired_tn_val)
# print(desired_test, desired_test_ratio, desired_tp_test, desired_tn_test)
#
# for aa in val_list: print(aa[0], aa[1])
# print('\n')
# for aa in test_list: print(aa[0], aa[1])


random.shuffle(train_list)
random.shuffle(val_list)
random.shuffle(test_list)

# for aa in val_list:
#     if 'accident' in aa[0]:
#         print(aa)

analyzed_dict_list_val = analyze_data_split(val_list)
analyzed_dict_list_val = analyze_data_split(test_list)
analyzed_dict_list_val = analyze_data_split(train_list)

# val_list = list(sorted(val_list, key=lambda x: (x[0], x[1])))
#
# for aa in val_list:
#     if 'accident' in aa[0]:
#         print(aa[0], aa[1])

for aa in train_list:
    print(aa[0], aa[1])
pdb.set_trace()

with open('./data/DeepAccident_data/train.txt', 'w') as f:
    for scenario_elem in train_list:
        # if scenario_elem[0] == "type1_subtype1_accident" and scenario_elem[1] == "Town03_type001_subtype0001_scenario00030":
        #     continue
        # if scenario_elem[0] == "type1_subtype1_accident" and scenario_elem[1] == "Town04_type001_subtype0001_scenario00030":
        #     continue
        f.write("%s %s\n" % (scenario_elem[0], scenario_elem[1]))

with open('./data/DeepAccident_data/val.txt', 'w') as f:
    for scenario_elem in val_list:
        # if scenario_elem[0] == "type1_subtype1_accident" and scenario_elem[1] == "Town03_type001_subtype0001_scenario00030":
        #     continue
        # if scenario_elem[0] == "type1_subtype1_accident" and scenario_elem[1] == "Town04_type001_subtype0001_scenario00030":
        #     continue
        f.write("%s %s\n" % (scenario_elem[0], scenario_elem[1]))

with open('./data/DeepAccident_data/test.txt', 'w') as f:
    for scenario_elem in test_list:
        # if scenario_elem[0] == "type1_subtype1_accident" and scenario_elem[1] == "Town03_type001_subtype0001_scenario00030":
        #     continue
        # if scenario_elem[0] == "type1_subtype1_accident" and scenario_elem[1] == "Town04_type001_subtype0001_scenario00030":
        #     continue
        f.write("%s %s\n" % (scenario_elem[0], scenario_elem[1]))
