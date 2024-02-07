#!/usr/bin/python
import os
import torch
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torch.nn.utils.rnn as rnn_utils
import pickle
import random
import json
from utils import setup_seed
from sklearn import preprocessing
from tptk.common.spatial_func import *
from tptk.common.trajectory import *
from tptk.common.grid import *

def prepare_camera_sequence_data(camera_traj, mat_padding_value, data_padding_value, min_len, max_len):

    # usr_id, week, weather, temperature  --> context data 每个轨迹都对应一条上下文信息（区别于stay query的上下文信息），包含上面四个特征
    # cllx，csys，cpys，minute,timestamp, cid, gid --> camera_traj_point 每个摄像头记录点包含这些信息
    # camera_traj_point_0, ..., camera_traj_point_n --> camera_traj_data 每条摄像头轨迹包含n个摄像头记录
    # camera_assign_mat --> 按照最长的序列扩充成 num_traj * max_length 的向量，有记录的位置为1，没有的为0

    gid_path_list = []
    sequence_list = []
    selected_idxs = []
    for i,traj in enumerate(camera_traj['new_mapping_traj']):
        gid_path = [pt[-1] for pt in traj]
        if len(gid_path) > max_len:
            gid_path = gid_path[0:max_len]
            traj = traj[0:max_len]
        elif len(gid_path) < min_len:
            continue
        selected_idxs.append(i)

        gid_path_list.append(torch.tensor(gid_path, dtype=torch.float32))
        sequence_list.append(torch.tensor(traj, dtype=torch.long)) # 包含时间戳，必须使用long型，padding使用-1，使用float会在最后几位丢失精度！

    camera_assign_mat = rnn_utils.pad_sequence(gid_path_list, padding_value=mat_padding_value,
                                               batch_first=True)  # num_traj * max_len

    camera_traj_data = rnn_utils.pad_sequence(sequence_list, padding_value=data_padding_value,
                                               batch_first=True)  # num_traj * max_len * 7

    context_data = torch.tensor(camera_traj.iloc[selected_idxs].drop(columns=['traj_id', 'new_mapping_traj']).values,dtype=torch.float)  # num_traj * 4
    return context_data, camera_traj_data, camera_assign_mat, selected_idxs

def prepare_label(label_map,num_traj,max_len,data_padding_value,selected_idxs): # label以pair为单位，data是以traj为单位
    max_stay_grid = max([len(stay_grids) for stay_grids in label_map.values()])
    stay_label = torch.full((num_traj, max_len, max_stay_grid), fill_value=data_padding_value)  # num_traj * max_len
    for key, value in label_map.items():
        for i, grid in enumerate(value):

            if key[1] > max_len - 1:
                pass
            else:
                stay_label[key[0], key[1], i] = grid

    stay_label = stay_label[selected_idxs]

    return stay_label

def prepare_candidate_regions(camera_traj_data, camera_map_intervted, dist_mat, candidate_threshold, num_traj, max_len, mat_padding_value,data_padding_value,max_candidate_grid):
    # 1.遍历所有pair
    # 2.计算卡口距离
    # 3.遍历dist_mat
    # 4.选出距离小于卡口距离*阈值的格子作为候选集
    candidate_region = torch.full((num_traj, max_len-1, max_candidate_grid), fill_value=mat_padding_value)
    # 对于pair来说，数量应该是max_len
    # cllx，csys，cpys，minute，timestamp，cid, gid
    # camera_traj_data.shape [36336, 20, 7]
    # todo return candidate region feature dist，angle，dist1，dist2
    camera_pair_feature = torch.full((num_traj, max_len-1, 3), fill_value=float('nan'))
    candidate_region_feature = torch.full((candidate_region.shape[0], candidate_region.shape[1], candidate_region.shape[2], 4), fill_value=float('nan'))
    camera_map = {value:key for value,key in camera_map_intervted.items()}
    for i in tqdm(range(camera_traj_data.shape[0]), desc='generate candidate region'):
        for j in range(camera_traj_data.shape[1]-1):
            t1 = int(camera_traj_data[i][j][-3])
            t2 = int(camera_traj_data[i][j+1][-3])
            cid1 = int(camera_traj_data[i][j][-2])
            cid2 = int(camera_traj_data[i][j+1][-2])
            gid1 = int(camera_traj_data[i][j][-1])
            gid2 = int(camera_traj_data[i][j+1][-1])

            if cid1 == data_padding_value or cid2== data_padding_value:
                continue
            lng2, lat2 = camera_map_intervted[cid2]
            lng1, lat1 = camera_map_intervted[cid1]
            kkgc_dist = haversine_distance(SPoint(lat2, lng2), SPoint(lat1, lng1))
            detour_threshold = candidate_threshold * kkgc_dist
            detour_dist = dist_mat[gid1][:] + dist_mat[:][gid2]
            OD_set = list(set([gid1, gid2]))
            candidate_set = np.where(detour_dist <= detour_threshold)[0]
            candidate_set = candidate_set[~np.isin(candidate_set, OD_set)] # 不在OD_set中的那部分
            # 构造候选集
            if len(candidate_set) > max_candidate_grid-len(OD_set):
                candidate_set = np.random.choice(candidate_set, size=max_candidate_grid-len(OD_set), replace=False, p=None) # 无放回抽样 max_candidate_grid 个
            candidate_set = list(candidate_set) + OD_set
            num_padding = max_candidate_grid - len(candidate_set)
            padding_candidate_set = candidate_set + [mat_padding_value] * num_padding
            candidate_region[i][j] = torch.tensor(padding_candidate_set, dtype=torch.long)

            # 构造pair feature
            pair_feature = [t2-t1, kkgc_dist, kkgc_dist/(t2-t1)]
            camera_pair_feature[i][j] = torch.tensor(pair_feature, dtype=torch.float)

            # 构造candidate region feature

            candidate_region_detour_dist = detour_dist[candidate_set].tolist() + [float('nan')]*num_padding
            candidate_region_dist1 = dist_mat[gid1][candidate_set].tolist() + [float('nan')]*num_padding
            candidate_region_dist2 = dist_mat[gid2][candidate_set].tolist() + [float('nan')]*num_padding
            candidate_region_angle = [angle(SPoint(camera_map[cid][1], camera_map[cid][0]),
                                                     SPoint(lat1, lng1),
                                                     SPoint(camera_map[cid][1],camera_map[cid][0]),
                                                     SPoint(lat2, lng2)) for cid in candidate_set] + [float('nan')]*num_padding
            candidate_region_detour_dist = torch.tensor(candidate_region_detour_dist, dtype=torch.float).unsqueeze(0)
            candidate_region_dist1 = torch.tensor(candidate_region_dist1, dtype=torch.float).unsqueeze(0)
            candidate_region_dist2 = torch.tensor(candidate_region_dist2, dtype=torch.float).unsqueeze(0)
            candidate_region_angle = torch.tensor(candidate_region_angle, dtype=torch.float).unsqueeze(0)
            region_feature = torch.cat([candidate_region_detour_dist, candidate_region_dist1, candidate_region_dist2, candidate_region_angle], dim=0)


            candidate_region_feature[i][j] = region_feature.t()

    # region A 占总pair的1.44%，占有stay的pair的7.5%
    return candidate_region, camera_pair_feature, candidate_region_feature

class Dataset(Dataset):
    def __init__(self, data):
        context_data, camera_traj_data, camera_assign_mat, stay_label, candidate_region, camera_pair_feature, candidate_region_feature = data
        self.context_data = torch.tensor(context_data, dtype=torch.float)
        self.camera_traj_data = torch.tensor(camera_traj_data, dtype=torch.float)
        self.camera_assign_mat = torch.tensor(camera_assign_mat, dtype=torch.long)
        self.stay_label = torch.tensor(stay_label, dtype=torch.long)
        self.candidate_region = torch.tensor(candidate_region, dtype=torch.long)
        self.camera_pair_feature = torch.tensor(camera_pair_feature, dtype=torch.float)
        self.candidate_region_feature = torch.tensor(candidate_region_feature, dtype=torch.float)

    def __len__(self):
        return self.context_data.shape[0]

    def __getitem__(self, idx):
        return (self.context_data[idx], self.camera_traj_data[idx], self.camera_assign_mat[idx], self.stay_label[idx], self.candidate_region[idx], self.camera_pair_feature[idx], self.candidate_region_feature[idx])

def get_sub_data(idxs,context_data,camera_traj_data,camera_assign_mat,stay_label,candidate_region,camera_pair_feature,candidate_region_feature):
    return [context_data[idxs], camera_traj_data[idxs], camera_assign_mat[idxs], stay_label[idxs], candidate_region[idxs], camera_pair_feature[idxs],candidate_region_feature[idxs]]

def split_dataset(split_ratio, context_data, camera_traj_data, camera_assign_mat, stay_label, candidate_region, camera_pair_feature,candidate_region_feature):
    train_ratio, val_ratio, test_ratio = np.array(split_ratio)/10
    assert train_ratio + val_ratio + test_ratio == 1
    idx = list(range(context_data.shape[0])) # num_traj
    random.shuffle(idx)
    length = len(idx)

    train_start_idx = 0
    train_end_idx = int(length * train_ratio)
    train_dataset = \
        get_sub_data(idx[train_start_idx:train_end_idx], context_data, camera_traj_data, camera_assign_mat, stay_label, candidate_region, camera_pair_feature,candidate_region_feature)

    val_start_idx = int(length * train_ratio)
    val_end_idx = int(length * (train_ratio+val_ratio))
    val_dataset = \
        get_sub_data(idx[val_start_idx:val_end_idx], context_data, camera_traj_data, camera_assign_mat, stay_label, candidate_region, camera_pair_feature,candidate_region_feature)

    test_start_idx = int(length * (train_ratio+val_ratio))
    test_end_idx = None
    test_dataset = \
        get_sub_data(idx[test_start_idx:test_end_idx], context_data, camera_traj_data, camera_assign_mat, stay_label, candidate_region, camera_pair_feature,candidate_region_feature)

    return train_dataset, val_dataset, test_dataset

def grid_standardization(grid_feature):
    # poi*16 rn*2 camera*1 row,col,lng,lat 都直接进行标准化
    # row col 参与标准化是因为23在数值上确实距离24更近，所以应该保留这种关系 作为连续值处理
    scaler = preprocessing.StandardScaler().fit(grid_feature)
    grid_feature = scaler.transform(grid_feature)
    return grid_feature

def context_standardization(train_dataset,val_dataset,test_dataset):
    # usr_id, week, weather, temperature
    # 不直接编码的离散值只有气温
    train_context_data_numberic = train_dataset[0][:, -1].reshape(-1, 1)
    val_context_data_numberic = val_dataset[0][:, -1].reshape(-1, 1)
    test_context_data_numberic = test_dataset[0][:, -1].reshape(-1, 1)

    scaler = preprocessing.StandardScaler().fit(train_context_data_numberic)

    train_context_data_numberic = scaler.transform(train_context_data_numberic).squeeze()
    val_context_data_numberic = scaler.transform(val_context_data_numberic).squeeze()
    test_context_data_numberic = scaler.transform(test_context_data_numberic).squeeze()

    train_dataset[0][:, -1] = torch.tensor(train_context_data_numberic, dtype=torch.float)
    val_dataset[0][:, -1] = torch.tensor(val_context_data_numberic, dtype=torch.float)
    test_dataset[0][:, -1] = torch.tensor(test_context_data_numberic, dtype=torch.float)
    return train_dataset, val_dataset, test_dataset

def pair_standardization(train_dataset,val_dataset,test_dataset):
    # distance, interval, speed 三个特征都进行编码
    # camera_pair_feature 进行标准化
    # train_dataset [context_data[idxs],camera_traj_data[idxs],camera_assign_mat[idxs],stay_label[idxs],candidate_region[idxs], camera_pair_feature[idxs]]

    num_pair_feature = 3
    train_camera_traj_data = train_dataset[1]
    max_len = train_camera_traj_data.shape[1]
    max_pair = max_len - 1

    train_camera_pair_feature_numberic = train_dataset[-2]
    val_camera_pair_feature_numberic = val_dataset[-2]
    test_camera_pair_feature_numberic = test_dataset[-2]

    num_train_data = train_camera_pair_feature_numberic.shape[0]
    num_val_data = val_camera_pair_feature_numberic.shape[0]
    num_test_data = test_camera_pair_feature_numberic.shape[0]

    # batch_size * (max_len-1) * 3  --> (batch_size*(max_len-1)) * 3
    # 对不同step的相同维度特征进行统一标准化
    train_camera_pair_feature_numberic = train_camera_pair_feature_numberic.reshape(-1, num_pair_feature)
    val_camera_pair_feature_numberic = val_camera_pair_feature_numberic.reshape(-1, num_pair_feature)
    test_camera_pair_feature_numberic = test_camera_pair_feature_numberic.reshape(-1, num_pair_feature)

    scaler = preprocessing.StandardScaler().fit(train_camera_pair_feature_numberic)

    train_camera_pair_feature_numberic = scaler.transform(train_camera_pair_feature_numberic)
    train_camera_pair_feature_numberic = train_camera_pair_feature_numberic.reshape(
        num_train_data, max_pair, num_pair_feature)

    val_camera_pair_feature_numberic = scaler.transform(val_camera_pair_feature_numberic)
    val_camera_pair_feature_numberic = val_camera_pair_feature_numberic.reshape(
        num_val_data, max_pair, num_pair_feature)

    test_camera_pair_feature_numberic = scaler.transform(test_camera_pair_feature_numberic)
    test_camera_pair_feature_numberic = test_camera_pair_feature_numberic.reshape(
        num_test_data, max_pair, num_pair_feature)

    train_dataset[-2] = torch.tensor(train_camera_pair_feature_numberic, dtype=torch.float)
    val_dataset[-2] = torch.tensor(val_camera_pair_feature_numberic, dtype=torch.float)
    test_dataset[-2] = torch.tensor(test_camera_pair_feature_numberic, dtype=torch.float)

    return train_dataset, val_dataset, test_dataset

def candidate_region_standardization(train_dataset, val_dataset, test_dataset): # 都是数值型

    def norm(data, mean, std):
        return (data-mean)/std

    train_candidate_region_feature_numberic = train_dataset[-1]
    val_candidate_region_feature_numberic = val_dataset[-1]
    test_candidate_region_feature_numberic = test_dataset[-1]


    train_candidate_region_detour_dist = train_candidate_region_feature_numberic[:, :, :, 0]
    train_candidate_region_dist1 = train_candidate_region_feature_numberic[:, :, :, 1]
    train_candidate_region_dist2 = train_candidate_region_feature_numberic[:, :, :, 2]
    train_candidate_region_angle = train_candidate_region_feature_numberic[:, :, :, 3]

    train_candidate_region_detour_dist_mean = np.nanmean(train_candidate_region_detour_dist)
    train_candidate_region_dist1_mean = np.nanmean(train_candidate_region_dist1)
    train_candidate_region_dist2_mean = np.nanmean(train_candidate_region_dist2)
    train_candidate_region_angle_mean = np.nanmean(train_candidate_region_angle)

    train_candidate_region_detour_dist_std = np.nanstd(train_candidate_region_detour_dist)
    train_candidate_region_dist1_std = np.nanstd(train_candidate_region_dist1)
    train_candidate_region_dist2_std = np.nanstd(train_candidate_region_dist2)
    train_candidate_region_angle_std = np.nanstd(train_candidate_region_angle)

    train_candidate_region_detour_dist = norm(train_candidate_region_detour_dist,
                                              train_candidate_region_detour_dist_mean,
                                              train_candidate_region_detour_dist_std)
    train_candidate_region_dist1 = norm(train_candidate_region_dist1,
                                        train_candidate_region_dist1_mean,
                                        train_candidate_region_dist1_std)
    train_candidate_region_dist2 = norm(train_candidate_region_dist2,
                                        train_candidate_region_dist2_mean,
                                        train_candidate_region_dist2_std)
    train_candidate_region_angle = norm(train_candidate_region_angle,
                                        train_candidate_region_angle_mean,
                                        train_candidate_region_angle_std)

    train_candidate_region_feature_numberic[:, :, :, 0] = train_candidate_region_detour_dist
    train_candidate_region_feature_numberic[:, :, :, 1] = train_candidate_region_dist1
    train_candidate_region_feature_numberic[:, :, :, 2] = train_candidate_region_dist2
    train_candidate_region_feature_numberic[:, :, :, 3] = train_candidate_region_angle

    train_dataset[-1] = train_candidate_region_feature_numberic

    val_candidate_region_detour_dist = val_candidate_region_feature_numberic[:, :, :, 0]
    val_candidate_region_dist1 = val_candidate_region_feature_numberic[:, :, :, 1]
    val_candidate_region_dist2 = val_candidate_region_feature_numberic[:, :, :, 2]
    val_candidate_region_angle = val_candidate_region_feature_numberic[:, :, :, 3]

    val_candidate_region_detour_dist = norm(val_candidate_region_detour_dist,
                                              train_candidate_region_detour_dist_mean,
                                              train_candidate_region_detour_dist_std)
    val_candidate_region_dist1 = norm(val_candidate_region_dist1,
                                        train_candidate_region_dist1_mean,
                                        train_candidate_region_dist1_std)
    val_candidate_region_dist2 = norm(val_candidate_region_dist2,
                                        train_candidate_region_dist2_mean,
                                        train_candidate_region_dist2_std)
    val_candidate_region_angle = norm(val_candidate_region_angle,
                                        train_candidate_region_angle_mean,
                                        train_candidate_region_angle_std)

    val_candidate_region_feature_numberic[:, :, :, 0] = val_candidate_region_detour_dist
    val_candidate_region_feature_numberic[:, :, :, 1] = val_candidate_region_dist1
    val_candidate_region_feature_numberic[:, :, :, 2] = val_candidate_region_dist2
    val_candidate_region_feature_numberic[:, :, :, 3] = val_candidate_region_angle

    val_dataset[-1] = val_candidate_region_feature_numberic


    test_candidate_region_detour_dist = test_candidate_region_feature_numberic[:, :, :, 0]
    test_candidate_region_dist1 = test_candidate_region_feature_numberic[:, :, :, 1]
    test_candidate_region_dist2 = test_candidate_region_feature_numberic[:, :, :, 2]
    test_candidate_region_angle = test_candidate_region_feature_numberic[:, :, :, 3]

    test_candidate_region_detour_dist = norm(test_candidate_region_detour_dist,
                                              train_candidate_region_detour_dist_mean,
                                              train_candidate_region_detour_dist_std)
    test_candidate_region_dist1 = norm(test_candidate_region_dist1,
                                        train_candidate_region_dist1_mean,
                                        train_candidate_region_dist1_std)
    test_candidate_region_dist2 = norm(test_candidate_region_dist2,
                                        train_candidate_region_dist2_mean,
                                        train_candidate_region_dist2_std)
    test_candidate_region_angle = norm(test_candidate_region_angle,
                                        train_candidate_region_angle_mean,
                                        train_candidate_region_angle_std)

    test_candidate_region_feature_numberic[:, :, :, 0] = test_candidate_region_detour_dist
    test_candidate_region_feature_numberic[:, :, :, 1] = test_candidate_region_dist1
    test_candidate_region_feature_numberic[:, :, :, 2] = test_candidate_region_dist2
    test_candidate_region_feature_numberic[:, :, :, 3] = test_candidate_region_angle

    test_dataset[-1] = test_candidate_region_feature_numberic

    return train_dataset, val_dataset, test_dataset

def get_loader(base_path, batch_size, num_worker, sequence_min_len, sequence_max_len, candidate_threshold, num_grid, max_candidate_grid):

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

    candidate_region, camera_pair_feature,candidate_region_feature = prepare_candidate_regions(camera_traj_data, camera_map_intervted, dist_mat, candidate_threshold, num_traj, max_len, num_grid, data_padding_value,max_candidate_grid)


    # print(candidate_region.shape)
    # print(candidate_region)

    # 先划分数据集，再进行标准化
    # 随机选择 60%的轨迹作为训练集，20%的轨迹作为测试集，20%的轨迹作为验证集
    split_ratio = [6, 2, 2]
    train_dataset, val_dataset, test_dataset = split_dataset(split_ratio, context_data, camera_traj_data, camera_assign_mat, stay_label, candidate_region,
                  camera_pair_feature,candidate_region_feature)

    train_dataset, val_dataset, test_dataset = context_standardization(train_dataset, val_dataset, test_dataset)

    train_dataset, val_dataset, test_dataset = pair_standardization(train_dataset, val_dataset, test_dataset)

    train_dataset, val_dataset, test_dataset = candidate_region_standardization(train_dataset, val_dataset, test_dataset)

    train_dataset = Dataset(train_dataset)
    val_dataset = Dataset(val_dataset)
    test_dataset = Dataset(test_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=True, num_workers=num_worker)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=True, num_workers=num_worker)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True, num_workers=num_worker)

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    config = json.load(open('config/region_C.json', 'r'))

    base_path = config['base_path']
    sequence_min_len = config['min_len']
    sequence_max_len = config['max_len']
    num_worker = config['num_worker']
    batch_size = config['batch_size']
    candidate_threshold = config['candidate_threshold']
    num_grid = config['num_grid']
    max_candidate_grid = config['max_candidate_grid']
    seed = config['random_seed']

    # 设置随机种子
    setup_seed(seed)

    train_loader, val_loader, test_loader = get_loader(base_path, batch_size, num_worker, sequence_min_len,
                                                       sequence_max_len, candidate_threshold, num_grid)

