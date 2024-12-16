import os
from dataloader import get_loader, grid_standardization
from utils import setup_seed
import numpy as np
import json
import torch
import pickle
import time
import argparse
from tqdm import tqdm
from metric import stay_detection_evaluation, stay_selection_evaluation

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

dev_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(dev_id)
torch.cuda.set_device(dev_id)

def evaluation(exp_path, model_name, dataloader, start_time, mode='eval'):
    # train parm
    region = config['region']
    base_path = config['base_path']
    max_candidate_grid = config['max_candidate_grid']
    num_grid = config['num_grid']

    grid_feature_path = os.path.join(base_path, 'grid_feature.pkl')
    grid_feature = pickle.load(open(grid_feature_path, 'rb'))
    grid_feature = grid_standardization(grid_feature)

    grid_feature_map = {gid: feature for gid, feature in enumerate(grid_feature)}

    model_path = os.path.join(exp_path, 'model', model_name)

    model = torch.load(model_path, map_location="cuda:{}".format(dev_id))['model'] 

    model.eval()

    binary_classification_label_list = []
    whether_stay_pred_list = []
    whether_stay_pred_prob_list = []

    where_stay_label_list = []
    where_stay_pred_list = []

    for idx, batch in tqdm(enumerate(dataloader)):
        context_data, camera_traj_data, camera_assign_mat, stay_label, \
        candidate_region, camera_pair_feature, candidate_region_feature = batch

        context_data, camera_traj_data, camera_assign_mat, camera_pair_feature = \
            context_data.cuda(), camera_traj_data.cuda(), camera_assign_mat.cuda(), camera_pair_feature.cuda()

        camera_pair_rep, binary_classification_label, \
        where_stay_pred, where_stay_label, pool_mask = model(camera_traj_data,
                                                                 camera_assign_mat,
                                                                 context_data,
                                                                 candidate_region,
                                                                 grid_feature_map,
                                                                 camera_pair_feature,
                                                                 candidate_region_feature,
                                                                 stay_label)
      
        whether_stay_pred = model.whether_stay_head(camera_pair_rep)
        whether_stay_pred = torch.sigmoid(whether_stay_pred)  
        whether_stay_pred_prob_list.append(whether_stay_pred.squeeze(-1).cpu().detach().numpy())

        whether_stay_pred = (whether_stay_pred >= 0.5).long().cpu()
        whether_stay_pred_list.append(whether_stay_pred.squeeze(-1))

        binary_classification_label = binary_classification_label.float().cpu()
        binary_classification_label_list.append(binary_classification_label)

        stay_pair_idx = torch.where(binary_classification_label==1)[0].numpy()

        real_stay_in_where_stay_pred = where_stay_pred[stay_pair_idx].squeeze(-1)  # num_pair*256*1
        real_stay_in_where_stay_label = where_stay_label[stay_pair_idx]  # num_pair*256

        real_stay_in_where_stay_pred = real_stay_in_where_stay_pred.cpu().detach().numpy()
        real_stay_in_where_stay_label = real_stay_in_where_stay_label.float().cpu().detach().numpy()

        where_stay_pred_list.append(real_stay_in_where_stay_pred)
        where_stay_label_list.append(real_stay_in_where_stay_label)

        if mode == 'detail':
            candidate_region_length = []
            for i in range(candidate_region.shape[0]):
                for j in range(candidate_region.shape[1]):
                    if camera_assign_mat[i][j] != num_grid and camera_assign_mat[i][j + 1] != num_grid:
                        candidate_region_set = candidate_region[i][j]
                        candidate_region_length.append(candidate_region_set[candidate_region_set != num_grid].shape[0])
            for i in stay_pair_idx:
                print(where_stay_pred[i][0:candidate_region_length[i]].reshape(-1))
                print(where_stay_label[i][0:candidate_region_length[i]])


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
    parser.add_argument('--region', default='')
    parser.add_argument('--exp_path', default='')
    opt = parser.parse_args()

    assert opt.exp_path.split('/')[-1].split('_')[1] == opt.region, '### region is not equal ###'

    exp_path = opt.exp_path
   

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

    setup_seed(seed)

    train_loader, val_loader, test_loader = get_loader(base_path, batch_size, num_worker, sequence_min_len,
                                                       sequence_max_len, candidate_threshold, num_grid, max_candidate_grid)



    start_time = time.time()
    # log_path = os.path.join(exp_path, 'evaluation')
    # sys.stdout = Logger(log_path, start_time, stream=sys.stdout)  # record log
    # sys.stderr = Logger(log_path, start_time, stream=sys.stderr)  # record error

    for _,_,files in os.walk(os.path.join(exp_path,'model')):
        for model_name in files:
            print(model_name)
            evaluation(exp_path, model_name, test_loader, start_time,mode='eval')

    