#!/usr/bin/python
import sys
sys.path.append('.')

from dataloader import *

def prepare_candidate_regions_framework(camera_traj_data, camera_map_intervted, dist_mat, candidate_threshold_type, candidate_threshold_value, num_traj, max_len, mat_padding_value,data_padding_value,max_candidate_grid):
    # 1.遍历所有pair
    # 2.计算卡口距离
    # 3.遍历dist_mat
    # 4.选出距离小于卡口距离*阈值的格子作为候选集
    print(candidate_threshold_type)
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

            
            if candidate_threshold_type in ['RInf','SHInf','VSHInf']:
                detour_threshold = candidate_threshold_value
            elif candidate_threshold_type in ['VHInf']:
                detour_threshold = max(candidate_threshold_value * (t2-t1), 0)
            else:
                detour_threshold = candidate_threshold_value * kkgc_dist


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


def get_loader(base_path, batch_size, num_worker, sequence_min_len, sequence_max_len, candidate_threshold_type, candidate_threshold_value, num_grid, max_candidate_grid):

    camera_sequence_path = os.path.join(base_path, 'sequence_data.pkl')
    label_path = os.path.join(base_path, 'label.pkl')
    dist_mat_path = os.path.join(base_path, 'dist_mat.pkl')
    camera_map_path = os.path.join(base_path, 'camera_map.pkl')

    num_samaple = 10000
    camera_traj = pickle.load(open(camera_sequence_path, 'rb'))
    camera_traj = camera_traj.head(num_samaple)  # 做一个采样提高实验效率
    
    label_map = pickle.load(open(label_path, 'rb'))
    tmp = {}
    for k,v in label_map.items():
        if k[0] < num_samaple:
            tmp[k]=v
    label_map = tmp

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

    candidate_region, camera_pair_feature,candidate_region_feature = prepare_candidate_regions_framework(camera_traj_data, camera_map_intervted, dist_mat, candidate_threshold_type, candidate_threshold_value, num_traj, max_len, num_grid, data_padding_value,max_candidate_grid)


    # print(candidate_region.shape)
    # print(candidate_region)

    # 先划分数据集，再进行标准化
    # 随机选择 60%的轨迹作为训练集，20%的轨迹作为测试集，20%的轨迹作为验证集
    split_ratio = [6, 2, 2]
    train_dataset, val_dataset, test_dataset = split_dataset(split_ratio, context_data, camera_traj_data, camera_assign_mat, stay_label, candidate_region,
                  camera_pair_feature,candidate_region_feature)

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

    candidate_threshold_type = 'VHInf'
    candidate_threshold_value = 3

    train_loader, val_loader, test_loader = get_loader(base_path, batch_size, num_worker, sequence_min_len,
                                                       sequence_max_len, candidate_threshold_type, candidate_threshold_value, num_grid, max_candidate_grid)

