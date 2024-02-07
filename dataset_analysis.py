#!/usr/bin/python
import os
import warnings

import torch

warnings.filterwarnings('ignore')
import pickle
import json
from utils import setup_seed
from dataloader import prepare_label, prepare_candidate_regions, prepare_camera_sequence_data

def  organize_dataset(config):
    region = config['region']
    base_path = config['base_path']
    sequence_min_len = config['min_len']
    sequence_max_len = config['max_len']
    candidate_threshold = config['candidate_threshold']
    num_grid = config['num_grid']
    max_candidate_grid = config['max_candidate_grid']

    seed = config['random_seed']
    setup_seed(seed)

    camera_sequence_path = os.path.join(base_path, 'sequence_data.pkl')
    label_path = os.path.join(base_path, 'label.pkl')
    dist_mat_path = os.path.join(base_path, 'dist_mat.pkl')
    camera_map_path = os.path.join(base_path, 'camera_map.pkl')

    camera_traj = pickle.load(open(camera_sequence_path, 'rb'))
    label_map = pickle.load(open(label_path, 'rb'))
    dist_mat = pickle.load(open(dist_mat_path, 'rb'))
    camera_map_intervted = pickle.load(open(camera_map_path, 'rb'))

    max_len = sequence_max_len
    min_len = sequence_min_len

    data_padding_value = -1
    context_data, camera_traj_data, camera_assign_mat, selected_idxs = prepare_camera_sequence_data(camera_traj, num_grid, data_padding_value, min_len, max_len)

    num_traj = context_data.shape[0]
    stay_label = prepare_label(label_map, num_traj, max_len, num_grid, selected_idxs) # stay_query_pair中可能存在多个驻留区域，所以shape 为 num_traj*20*5

    # min_len 会筛选出长度小于它的camera_traj，新的camera_traj_idx是selected_idxs
    # 每个对应的context_data, camera_traj_data, camera_assign_mat, stay_label 的一行是对应的

    candidate_region, camera_pair_feature = prepare_candidate_regions(camera_traj_data, camera_map_intervted, dist_mat, candidate_threshold, num_traj, max_len, num_grid, data_padding_value, max_candidate_grid)

    num_traj = context_data.shape[0] # num_traj
    num_pair = camera_pair_feature.shape[0] #
    num_stay_pair = (camera_pair_feature.numel() - torch.isnan(camera_pair_feature).long().sum())/3

    # 平均候选个数
    num_each_candidate_region = (candidate_region != num_grid).long().sum(-1)
    mean_candidate_region = num_each_candidate_region[num_each_candidate_region!=0].float().mean()

    num_each_stay_region = (stay_label != num_grid).long().sum(-1)
    mean_candidate_region_in_stay_pair = num_each_candidate_region[num_each_stay_region[:, 0:-1]!=0].float().mean()

    print('region: {}'.format(region))
    print('num_traj: {}'.format(num_traj))
    print('num_pair: {}'.format(num_pair))
    print('num_stay_pair: {}'.format(num_stay_pair))
    print('average count of candidate region in unpadding pair: {}'.format(mean_candidate_region))
    print('average count of candidate region in stay unpadding pair: {}'.format(mean_candidate_region_in_stay_pair))


if __name__ == '__main__':
    config_A = json.load(open('config/region_A.json', 'r'))
    organize_dataset(config_A)

    config_B = json.load(open('config/region_B.json', 'r'))
    organize_dataset(config_B)

    config_C = json.load(open('config/region_C.json', 'r'))
    organize_dataset(config_C)

# region: A
# num_traj: 36336
# num_pair: 36336
# num_stay_pair: 254069.0
# average count of candidate region in unpadding pair: 39.14462661743164
# average count of candidate region in stay unpadding pair: 54.76809310913086
# generate candidate region: 100%|██████████| 61946/61946 [01:16<00:00, 812.73it/s]

# region: B
# num_traj: 61946
# num_pair: 61946
# num_stay_pair: 556783.0
# average count of candidate region in unpadding pair: 38.7722053527832
# average count of candidate region in stay unpadding pair: 79.41786193847656
# generate candidate region: 100%|██████████| 13838/13838 [00:15<00:00, 876.82it/s]

# region: C
# num_traj: 13838
# num_pair: 13838
# num_stay_pair: 117986.0
# average count of candidate region in unpadding pair: 25.508907318115234
# average count of candidate region in stay unpadding pair: 43.613868713378906