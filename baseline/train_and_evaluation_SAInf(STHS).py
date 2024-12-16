import os
from framework_dataloader import get_loader
from utils import setup_seed
import numpy as np
import json
from datetime import datetime
import torch
import torch.nn as nn
import time,datetime
import argparse
from tqdm import tqdm
from metric import hitK
from evaluation_SAInf import stay_detection_evaluation,stay_selection_evaluation
from frameworks import framework
from SAInf import Detection_Threshold_Estimation

# 直接调用framework
# speed_threshold 使用KS test进行估计
# 候选区域通过离心率与经验参数动态计算
# 更改grid_freq的大小，原来是num_grid+1,现在是num_grid*24+1
def timestamp2hour(timestamp):
    if timestamp== -1:
        return -1
    else:
        return datetime.datetime.fromtimestamp(timestamp).hour
    
def get_pair_speed_and_label(camera_assign_mat, camera_pair_feature, stay_label):
    binary_classification_label_list = []
    camera_pair_speed_list = []
    for i in range(camera_assign_mat.shape[0]):
        for j in range(camera_assign_mat.shape[1] - 1):
            if camera_assign_mat[i][j] == num_grid or camera_assign_mat[i][j + 1] == num_grid:
                continue
            else:
                # camera_assign_mat中第 j 和 j+1 的记录对 对应的是camera_pair_feature 中的第j个
                camera_pair_speed = camera_pair_feature[i][j][-1]
                camera_pair_speed_list.append(camera_pair_speed)

            num_stay_grid = (stay_label[i][j] != num_grid).long().sum().item()
            if num_stay_grid == 0:
                binary_classification_label_list.append(0)
            else:
                binary_classification_label_list.append(1)

    binary_classification_label = torch.tensor(binary_classification_label_list)
    camera_pair_speed = torch.tensor(camera_pair_speed_list)
    return binary_classification_label, camera_pair_speed


def get_where_stay_label(candidate_region, stay_label, camera_traj_data, num_grid):
    
    camera_traj_timestamp = camera_traj_data[:,:,-3].long()
    camera_traj_hour = torch.tensor([[timestamp2hour(timestamp) for timestamp in row]for row in camera_traj_timestamp])
    
    pair_candidate_region_list = []
    where_stay_label_list = []
    for i in range(candidate_region.shape[0]):  # -1为padding位
        for j in range(candidate_region.shape[1]):
            candidate_region_set = candidate_region[i][j]
            if (candidate_region_set != num_grid).long().sum().item() == 0:  # 在构造candidate_region时，padding的位置的候选集全部都为num_grid
                continue  # padding pair位置pass
            else:  # 非padding pair
                hour = camera_traj_hour[i][j]
                update_candidate_region_set = torch.where(candidate_region_set==num_grid, num_grid*24, candidate_region_set*24+hour)
                pair_candidate_region_list.append(update_candidate_region_set.unsqueeze(0))

                candidate_region_set = candidate_region_set[candidate_region_set != num_grid]
                stay_region = stay_label[i][j]
                stay_region = stay_region[stay_region != num_grid]

                label_idx = torch.where((candidate_region_set[..., None] == stay_region).any(-1))[0]
                label = torch.zeros(candidate_region.shape[-1])
                label[label_idx] = 1

                where_stay_label_list.append(label.unsqueeze(0))

        
    where_stay_label = torch.cat(where_stay_label_list, dim=0)
    pair_candidate_region = torch.cat(pair_candidate_region_list, dim=0)

    return pair_candidate_region, where_stay_label

# 原来的 0-num_grid-1 --> 0-23,24-47,48-71,...(num_grid-1)*24-1
# 
def train(framework_type, num_grid, dataloader):
    cnt = 0
    if framework_type == 'RInf':
        return torch.tensor(num_grid*[1.0] + [-1.0])
    elif framework_type in ['SHInf','VHInf','VSHInf','SAInf']:
        freq = (num_grid*24+1)*[0.0]
        stay_label = dataloader.dataset.stay_label
        camera_traj_data = dataloader.dataset.camera_traj_data
        camera_traj_timestamp = camera_traj_data[:,:,-3].long()
        camera_traj_hour = torch.tensor([[timestamp2hour(timestamp) for timestamp in row]for row in camera_traj_timestamp])
        for i in tqdm(range(stay_label.shape[0]), desc='compute_freq'):
            for j in range(stay_label.shape[1]-1):
                stay_grid_list = stay_label[i][j]
                hour = camera_traj_hour[i][j]
                if (stay_grid_list != num_grid).long().sum() !=0 : # 不是padding的pair
                    for grid in stay_grid_list:
                        if grid != num_grid:
                            freq[grid*24+hour] += 1
                            cnt+=1
        freq[-1] = -1.0
        return torch.tensor(freq)           

def evaluation(model, dataloader, start_time):
    
    binary_classification_label_list = []
    whether_stay_pred_list = []
    whether_stay_pred_prob_list = []

    where_stay_label_list = []
    where_stay_pred_list = []

    for idx, batch in enumerate(dataloader):
        _, camera_traj_data, camera_assign_mat, stay_label, \
        candidate_region, camera_pair_feature, _ = batch

        binary_classification_label, camera_pair_speed = get_pair_speed_and_label(camera_assign_mat,camera_pair_feature,stay_label)
        pair_candidate_region, where_stay_label = get_where_stay_label(candidate_region, stay_label, camera_traj_data, num_grid)

        # camera_traj_data 应该拉平
        # 然后用其更新pair_candidate_region的id值

        whether_stay_pred, where_stay_pred = model.stay_area_inference(camera_pair_speed, pair_candidate_region)  

        whether_stay_pred = whether_stay_pred.long()
        binary_classification_label = binary_classification_label.float()

        whether_stay_pred_prob_list.append(whether_stay_pred.squeeze(-1).numpy())
        whether_stay_pred_list.append(whether_stay_pred.squeeze(-1).numpy())
        binary_classification_label_list.append(binary_classification_label)

        stay_pair_idx = torch.where(binary_classification_label==1)[0].numpy()

        real_stay_in_where_stay_pred = where_stay_pred[stay_pair_idx].squeeze(-1)  # num_pair*256*1
        real_stay_in_where_stay_label = where_stay_label[stay_pair_idx]  # num_pair*256

        real_stay_in_where_stay_pred = real_stay_in_where_stay_pred.detach().numpy()
        real_stay_in_where_stay_label = real_stay_in_where_stay_label.float().numpy()

        where_stay_pred_list.append(real_stay_in_where_stay_pred)
        where_stay_label_list.append(real_stay_in_where_stay_label)

    binary_classification_label = np.concatenate(binary_classification_label_list)
    whether_stay_pred = np.concatenate(whether_stay_pred_list)
    whether_stay_pred_prob = np.concatenate(whether_stay_pred_prob_list)  

    where_stay_label = np.concatenate(where_stay_label_list)
    where_stay_pred = np.concatenate(where_stay_pred_list)

    acc, auc, precision, recall, f1 = stay_detection_evaluation(binary_classification_label, whether_stay_pred, whether_stay_pred_prob)
    print('acc:{:.4f} auc:{:.4f}'.format(acc, auc))
    print('precision:{:.4f} recall:{:.4f} f1:{:.4f}'.format(precision, recall, f1))

    hit1, hit3, hit5 = stay_selection_evaluation(where_stay_label, where_stay_pred)

    print('hit@1:{:.4f} hit@3:{:.4f} hit@5:{:.4f} SUM:{:.4f}'.format(hit1, hit3, hit5, hit1 + hit3 + hit5))
           
           
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--region', default='',help='A | B | C ')
    opt = parser.parse_args()

    config = json.load(open('config/region_{}.json'.format(opt.region), 'r'))

    base_path = config['base_path']
    batch_size = config['batch_size']
    sequence_min_len = config['min_len']
    sequence_max_len = config['max_len']
    num_worker = config['num_worker']
    candidate_threshold = config['candidate_threshold']
    num_grid = config['num_grid']
    max_candidate_grid = config['max_candidate_grid']
    seed = config['random_seed']

    # 设置随机种子
    setup_seed(seed)

    framework_type = 'SAInf'

    candidate_threshold_type = framework_type
    candidate_threshold_value = config['candidate_threshold']

    train_loader, val_loader, test_loader = get_loader(base_path, batch_size, num_worker, sequence_min_len,
                                                       sequence_max_len, candidate_threshold_type, candidate_threshold_value, num_grid,
                                                       max_candidate_grid)
    pair_speed = train_loader.dataset.camera_pair_feature[:,:,-1] # num_traj * 19 *3
    stay_label = train_loader.dataset.stay_label # num_traj * 20 * 3

    stay_distribution = [] 
    unstay_distribution = []
    for i in tqdm(range(stay_label.shape[0]), desc='estimate speed threshold'):
        for j in range(stay_label.shape[1]-1):
            speed = pair_speed[i][j] # 有负数，因为标准化过
            if torch.isnan(speed):
                pass
            else:
                if (stay_label[i][j] != num_grid).long().sum() !=0 :
                    stay_distribution.append(round(speed.item(),4))
                else:
                    unstay_distribution.append(round(speed.item(), 4))


    Estimator = Detection_Threshold_Estimation(stay_distribution, unstay_distribution, 10000)
    speed_threshold = Estimator.estimate_threshold()

    
    grid_freq_weight = train(framework_type, num_grid, train_loader)

    model = framework(num_grid*24, speed_threshold, grid_freq_weight)

    start_time = time.time()
    evaluation(model, test_loader, start_time)



# dataset A
# acc:0.8324 auc:0.8833
# precision:0.5078 recall:0.9611 f1:0.6645
# hit@1:0.3115 hit@3:0.4868 hit@5:0.5723 SUM:1.3706

# dataset B
# acc:0.7873 auc:0.8568
# precision:0.3606 recall:0.9489 f1:0.5226
# hit@1:0.2815 hit@3:0.4257 hit@5:0.5044 SUM:1.2115   

# dataset C
# acc:0.8224 auc:0.8755
# precision:0.4482 recall:0.9504 f1:0.6091
# hit@1:0.3474 hit@3:0.5189 hit@5:0.6083 SUM:1.4746

