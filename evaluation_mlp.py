import os
from dataloader import get_loader, grid_standardization
from utils import setup_seed
import numpy as np
import json
import torch
import pickle
import time
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from metric import hitK
from line_profiler import LineProfiler

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

dev_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(dev_id)
torch.cuda.set_device(dev_id)

def stay_detection_evaluation(binary_classification_label, whether_stay_pred, whether_stay_pred_prob):

    acc = accuracy_score(binary_classification_label, whether_stay_pred)
    auc = roc_auc_score(binary_classification_label, whether_stay_pred_prob)

    report = classification_report(binary_classification_label, whether_stay_pred, output_dict=True)['1.0']
    precision = report['precision']
    recall = report['recall']
    f1 = report['f1-score']
    return acc, auc, precision, recall, f1

def stay_selection_evaluation(where_stay_label, where_stay_pred):
    hit1 = hitK(where_stay_pred, where_stay_label, 1)
    hit3 = hitK(where_stay_pred, where_stay_label, 3)
    hit5 = hitK(where_stay_pred, where_stay_label, 5)
    return hit1, hit3, hit5


def evaluation(exp_path, model_name, dataloader, start_time, mode='eval'):
    # train parm
    region = config['region']
    base_path = config['base_path']
    max_candidate_grid = config['max_candidate_grid']
    num_grid = config['num_grid']

    edge_index_path = os.path.join(base_path, 'adj.pkl')
    edge_index = pickle.load(open(edge_index_path, 'rb'))

    grid_feature_path = os.path.join(base_path, 'grid_feature.pkl')
    grid_feature = pickle.load(open(grid_feature_path, 'rb'))
    grid_feature = grid_standardization(grid_feature)
 
    grid_feature_map = {gid: feature for gid, feature in enumerate(grid_feature)}

    model_path = os.path.join(exp_path, 'model', model_name)

    model = torch.load(model_path, map_location="cuda:{}".format(dev_id))['model']  # model.to()包含inplace操作，不需要对象承接

    model.eval()

    binary_classification_label_list = []
    whether_stay_pred_list = []
    whether_stay_pred_prob_list = []


    # unpadding_pair_num = 0
    # stay_pair_num = 0
    # candidate_region_num_list = []
    # detect_stay_pair_num = 0
    # detect_real_stay_pair_num = 0

    for idx, batch in tqdm(enumerate(dataloader)):
        context_data, _, camera_assign_mat, stay_label, \
        _, camera_pair_feature,_ = batch

        context_data, camera_assign_mat, camera_pair_feature = \
        context_data.cuda(), camera_assign_mat.cuda(), camera_pair_feature.cuda()
            
        batch_binary_classification_label_list = []
        flatten_camera_pair_feature_list = []
        for i in range(camera_assign_mat.shape[0]):
            for j in range(camera_assign_mat.shape[1] - 1):
                if camera_assign_mat[i][j] == num_grid or camera_assign_mat[i][j + 1] == num_grid:
                    continue
                else:
                    flatten_camera_pair_feature_list.append(camera_pair_feature[i][j].unsqueeze(0))

                num_stay_grid = (stay_label[i][j] != num_grid).long().sum().item()
                if num_stay_grid == 0:
                    batch_binary_classification_label_list.append(0)
                else:
                    batch_binary_classification_label_list.append(1)

        camera_pair_feature = torch.cat(flatten_camera_pair_feature_list, dim=0)
        binary_classification_label = torch.tensor(batch_binary_classification_label_list)
            
        whether_stay_pred = model(camera_pair_feature)

        whether_stay_pred_prob_list.append(whether_stay_pred.squeeze(-1).cpu().detach().numpy())

        whether_stay_pred = (whether_stay_pred >= 0.5).long().cpu()
        whether_stay_pred_list.append(whether_stay_pred.squeeze(-1))

        binary_classification_label = binary_classification_label.float().cpu()
        binary_classification_label_list.append(binary_classification_label)



    binary_classification_label = np.concatenate(binary_classification_label_list)
    whether_stay_pred = np.concatenate(whether_stay_pred_list)
    whether_stay_pred_prob = np.concatenate(whether_stay_pred_prob_list)

    acc, auc, precision, recall, f1 = stay_detection_evaluation(binary_classification_label, whether_stay_pred, whether_stay_pred_prob)
    print('acc:{:.4f} auc:{:.4f}'.format(acc, auc))
    print('precision:{:.4f} recall:{:.4f} f1:{:.4f}'.format(precision, recall, f1))

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

    train_loader, val_loader, test_loader = get_loader(base_path, batch_size, num_worker, sequence_min_len,
                                                       sequence_max_len, candidate_threshold, num_grid, max_candidate_grid)

    exp_path = './exp/MLP_C_240312024910'
    model_name = 'MLP_C_20_240312024910_19.pt'

    start_time = time.time()
    # log_path = os.path.join(exp_path, 'evaluation')
    # sys.stdout = Logger(log_path, start_time, stream=sys.stdout)  # record log
    # sys.stderr = Logger(log_path, start_time, stream=sys.stderr)  # record error
    evaluation(exp_path, model_name, test_loader, start_time,mode='eval')

