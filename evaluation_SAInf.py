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
from SAInf import Detection_Threshold_Estimation

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

dev_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(dev_id)
torch.cuda.set_device(dev_id)

def stay_detection_evaluation(binary_classification_label, whether_stay_pred, whether_stay_pred_prob):

    # acc 和 auc 针对二分类而言
    acc = accuracy_score(binary_classification_label, whether_stay_pred)
    auc = roc_auc_score(binary_classification_label, whether_stay_pred_prob)
    # precision, recall, f1 针对 label=1 的类别而言
    # 计算全部类的性能
    #
    # 计算某个类的性能
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


def evaluation(exp_path, model_name, dataloader, start_time, threshold, mode='eval'):
    # train parm
    region = config['region']
    base_path = config['base_path']
    max_candidate_grid = config['max_candidate_grid']
    num_grid = config['num_grid']


    grid_feature_path = os.path.join(base_path, 'grid_feature.pkl')
    grid_feature = pickle.load(open(grid_feature_path, 'rb'))
    grid_feature = grid_standardization(grid_feature)
    # 转换成字典提高候选集构造的速度
    grid_feature_map = {gid: feature for gid, feature in enumerate(grid_feature)}

    model_path = os.path.join(exp_path, 'model', model_name)

    model = torch.load(model_path, map_location="cuda:{}".format(dev_id))['model']  # model.to()包含inplace操作，不需要对象承接

    model.eval()

    binary_classification_label_list = []
    whether_stay_pred_list = []
    whether_stay_pred_prob_list = []

    where_stay_label_list = []
    where_stay_pred_list = []


    # 分析数据情况
    # unpadding_pair_num = 0
    # stay_pair_num = 0
    # candidate_region_num_list = []
    # detect_stay_pair_num = 0
    # detect_real_stay_pair_num = 0

    for idx, batch in tqdm(enumerate(dataloader)):
        context_data, camera_traj_data, camera_assign_mat, stay_label, \
        candidate_region, camera_pair_feature, candidate_region_feature = batch

        context_data, camera_traj_data, camera_assign_mat, camera_pair_feature = \
            context_data.cuda(), camera_traj_data.cuda(), camera_assign_mat.cuda(), camera_pair_feature.cuda()

        camera_pair_speed, binary_classification_label, \
        stay_query_pair_rep, where_stay_label, pool_mask = model(camera_traj_data,
                                                                 camera_assign_mat,
                                                                 context_data,
                                                                 candidate_region,
                                                                 grid_feature_map,
                                                                 camera_pair_feature,
                                                                 candidate_region_feature,
                                                                 stay_label)

        whether_stay_pred = model.stay_evenet_detection(camera_pair_speed).long()
        binary_classification_label = binary_classification_label.float().cpu()

        # 驻留事件检测

        whether_stay_pred_prob_list.append(whether_stay_pred.squeeze(-1).cpu().detach().numpy())
        whether_stay_pred_list.append(whether_stay_pred.squeeze(-1).cpu().detach().numpy())
        binary_classification_label_list.append(binary_classification_label)

        # 驻留区域选择
        where_stay_pred = model.where_stay_head(stay_query_pair_rep)
        where_stay_pred = torch.sigmoid(where_stay_pred)  # 约束值域为0-1
        where_stay_pred = where_stay_pred * pool_mask # padding位置为0

        stay_pair_idx = torch.where(binary_classification_label==1)[0].numpy()

        # 计算metric
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

    pair_speed = train_loader.dataset.camera_pair_feature[:, :, -1]  # num_traj * 19 *3
    stay_label = train_loader.dataset.stay_label  # num_traj * 20 * 3

    stay_distribution = []
    unstay_distribution = []
    for i in tqdm(range(stay_label.shape[0]), desc='estimate speed threshold'):
        for j in range(stay_label.shape[1] - 1):
            speed = pair_speed[i][j]  # 有负数，因为标准化过
            if torch.isnan(speed):
                pass
            else:
                if (stay_label[i][j] != num_grid).long().sum() != 0:
                    stay_distribution.append(round(speed.item(), 4))
                else:
                    unstay_distribution.append(round(speed.item(), 4))

    Estimator = Detection_Threshold_Estimation(stay_distribution, unstay_distribution, 10000)
    threshold = Estimator.estimate_threshold()


    exp_path = 'D:/Project/SAInf++/exp/SAInf_C_240206054926'
    model_name = 'SAInf_C_20_240206054926_19.pt'

    start_time = time.time()
    # log_path = os.path.join(exp_path, 'evaluation')
    # sys.stdout = Logger(log_path, start_time, stream=sys.stdout)  # record log
    # sys.stderr = Logger(log_path, start_time, stream=sys.stderr)  # record error
    evaluation(exp_path, model_name, test_loader, start_time, threshold) # mode='detail'


# exp_path = 'D:/Project/SAInf++/exp/SAInf_C_240203072253'
# model_name = 'SAInf_C_20_240203072253_19.pt'
# acc:0.8135 auc:0.8721
# precision:0.4346 recall:0.9546 f1:0.5973
# hit@1:0.3414 hit@3:0.5359 hit@5:0.6225 SUM:1.4998

# exp_path = 'D:/Project/SAInf++/exp/SAInf_C_240203085112'
# model_name = 'SAInf_C_20_240203085112_19.pt'
# acc:0.8135 auc:0.8721
# precision:0.4346 recall:0.9546 f1:0.5973
# hit@1:0.3571 hit@3:0.5343 hit@5:0.6362 SUM:1.5276


# exp_path = 'D:/Project/SAInf++/exp/SAInf_A_240203182410'
# model_name = 'SAInf_A_20_240203182410_19.pt'
# acc:0.8375 auc:0.8860
# precision:0.5259 recall:0.9616 f1:0.6800
# hit@1:0.2594 hit@3:0.4308 hit@5:0.5287 SUM:1.2189

# exp_path = 'D:/Project/SAInf++/exp/SAInf_B_240203222705'
# model_name = 'SAInf_B_20_240203222705_19.pt'
# acc:0.7961 auc:0.8619
# precision:0.3960 recall:0.9523 f1:0.5594
# hit@1:0.2640 hit@3:0.4381 hit@5:0.5292 SUM:1.2313


# 不使用车辆车身车辆类型 embedding
# exp_path = 'D:/Project/SAInf++/exp/SAInf_C_240206034117'
# model_name = 'SAInf_C_20_240206034117_19.pt'
# acc:0.8135 auc:0.8721
# precision:0.4346 recall:0.9546 f1:0.5973
# hit@1:0.4184 hit@3:0.6490 hit@5:0.7605 SUM:1.8279
# hit@1:0.4069 hit@3:0.6682 hit@5:0.7729 SUM:1.8480


# 车牌车身车辆类型 embedding
# exp_path = 'D:/Project/SAInf++/exp/SAInf_C_240206054926'
# model_name = 'SAInf_C_20_240206054926_19.pt'
# acc:0.8135 auc:0.8721
# precision:0.4346 recall:0.9546 f1:0.5973
# hit@1:0.4158 hit@3:0.6544 hit@5:0.7732 SUM:1.8435
