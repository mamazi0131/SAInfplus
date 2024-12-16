import os
from framework_dataloader import get_loader
from utils import setup_seed
import numpy as np
import json
from datetime import datetime
import torch
import torch.nn as nn
import time
from tqdm import tqdm
from metric import hitK
from evaluation_SAInf import stay_detection_evaluation,stay_selection_evaluation
from frameworks import framework

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

def get_where_stay_label(candidate_region, stay_label, num_grid):
    pair_candidate_region_list = []
    where_stay_label_list = []
    for i in range(candidate_region.shape[0]):  # -1为padding位
        for j in range(candidate_region.shape[1]):
            candidate_region_set = candidate_region[i][j]
            if (candidate_region_set != num_grid).long().sum().item() == 0:  # 在构造candidate_region时，padding的位置的候选集全部都为num_grid
                continue  # padding pair位置pass
            else:  # 非padding pair
                pair_candidate_region_list.append(candidate_region_set.unsqueeze(0))

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

def train(framework_type, num_grid, dataloader):
    if framework_type == 'RInf':
        return torch.tensor(num_grid*[1.0] + [-1.0])
    elif framework_type in ['SHInf','VHInf','VSHInf','SAInf']:
        freq = (num_grid+1)*[0.0]
        stay_label = dataloader.dataset.stay_label
        for i in tqdm(range(stay_label.shape[0]), desc='compute_freq'):
            for j in range(stay_label.shape[1]-1):
                stay_grid_list = stay_label[i][j]
                if (stay_grid_list != num_grid).long().sum() !=0 : # 不是padding的pair
                    for grid in stay_grid_list: 
                        freq[grid] += 1
        freq[-1] = -1.0
        return torch.tensor(freq)           

def evaluation(model, dataloader, start_time):
    
    binary_classification_label_list = []
    whether_stay_pred_list = []
    whether_stay_pred_prob_list = []

    where_stay_label_list = []
    where_stay_pred_list = []

    for idx, batch in enumerate(dataloader):
        _, _, camera_assign_mat, stay_label, \
        candidate_region, camera_pair_feature, _ = batch

        binary_classification_label, camera_pair_speed = get_pair_speed_and_label(camera_assign_mat,camera_pair_feature,stay_label)
        pair_candidate_region, where_stay_label = get_where_stay_label(candidate_region, stay_label, num_grid)

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
    config = json.load(open('config/region_C.json', 'r'))

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

    framework_type = 'VHInf'

    candidate_threshold_type = framework_type
    candidate_threshold_value = 3

    train_loader, val_loader, test_loader = get_loader(base_path, batch_size, num_worker, sequence_min_len,
                                                       sequence_max_len, candidate_threshold_type, candidate_threshold_value, num_grid,
                                                       max_candidate_grid)

    speed_threshold = 3
    # 对于framework，train的过程是获得grid_freq_weight
    grid_freq_weight = train(framework_type, num_grid, train_loader)

    model = framework(num_grid, speed_threshold, grid_freq_weight)

    start_time = time.time()
    evaluation(model, test_loader, start_time)

# A
# acc:0.8523 auc:0.8406
# precision:0.5483 recall:0.8226 f1:0.6580
# hit@1:0.2608 hit@3:0.4512 hit@5:0.5448 SUM:1.2568

# B
# acc:0.8029 auc:0.8456
# precision:0.3742 recall:0.9021 f1:0.5290
# hit@1:0.2159 hit@3:0.3907 hit@5:0.4786 SUM:1.0852

# C
# acc:0.8381 auc:0.8706
# precision:0.4713 recall:0.9163 f1:0.6225
# hit@1:0.2412 hit@3:0.4434 hit@5:0.5369 SUM:1.2215