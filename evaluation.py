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


def evaluation(exp_path, model_name, dataloader, start_time):
    # train parm
    region = config['region']
    base_path = config['base_path']
    max_candidate_grid = config['max_candidate_grid']
    num_grid = config['num_grid']

    edge_index_path = os.path.join(base_path, 'adj.pkl')
    edge_index = pickle.load(open(edge_index_path, 'rb'))

    # todo 如果在dataloader中标准化，则在dataloader中处理。如果在这里处理，就在model中的处理
    grid_feature_path = os.path.join(base_path, 'grid_feature.pkl')
    grid_feature = pickle.load(open(grid_feature_path, 'rb'))

    grid_feature = grid_standardization(grid_feature)

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
        context_data, camera_traj_data, camera_assign_mat, stay_label, candidate_region, camera_pair_feature = batch

        context_data, camera_traj_data, camera_assign_mat, stay_label, candidate_region, camera_pair_feature =\
        context_data.cuda(), camera_traj_data.cuda(), camera_assign_mat.cuda(), stay_label.cuda(), candidate_region.cuda(), camera_pair_feature.cuda()


        camera_pair_rep, binary_classification_label, stay_query_pair_rep, stay_label, pool_mask = model(camera_traj_data,
                                                                                                  camera_assign_mat,
                                                                                                  context_data,
                                                                                                  candidate_region,
                                                                                                  grid_feature,
                                                                                                  camera_pair_feature,
                                                                                                  stay_label)

        whether_stay_pred = model.whether_stay_head(camera_pair_rep)
        whether_stay_pred = torch.sigmoid(whether_stay_pred)  # 约束值域为0-1
        whether_stay_pred_prob_list.append(whether_stay_pred.squeeze(-1).cpu().detach().numpy())

        whether_stay_pred = (whether_stay_pred >= 0.5).long().cpu()


        binary_classification_label = binary_classification_label.float().cpu()

        binary_classification_label_list.append(binary_classification_label)

        whether_stay_pred_list.append(whether_stay_pred.squeeze(-1))


        # 计算全部类的性能
        # acc = accuracy_score(binary_classification_label, whether_stay_pred)
        # 计算某个类的性能
        # classification_report(binary_classification_label, whether_stay_pred, output_dict=True)['1.0']

        where_stay_pred = model.where_stay_head(stay_query_pair_rep)
        where_stay_pred = torch.sigmoid(where_stay_pred)  # 约束值域为0-1
        where_stay_pred = where_stay_pred * pool_mask # padding位置为0

        # 1. 找到binary_classification_label中有驻留的pair
        detected_pair_idx = torch.where(whether_stay_pred > 0.5)[0].numpy()
        undetected_pair_idx = torch.where(whether_stay_pred <= 0.5)[0].numpy()
        stay_pair_idx = torch.where(binary_classification_label==1)[0].numpy()

        # 2. stay_pair_idx 计算loss
        # 3. stay_pair_idx 计算metric

        # 可复用
        num_pair = stay_query_pair_rep.shape[0]
        where_stay_label = torch.full((num_pair, max_candidate_grid), fill_value=0)
        cnt_1 = 0 # 所有pair的数量
        cnt_2 = 0 # 有驻留的pair的数量
        candidate_region_length = []
        for i in range(candidate_region.shape[0]):
            for j in range(candidate_region.shape[1]):
                if camera_assign_mat[i][j]!=num_grid and camera_assign_mat[i][j+1]!=num_grid:
                    candidate_region_set = candidate_region[i][j]
                    candidate_region_length.append(candidate_region_set[candidate_region_set != num_grid].shape[0])
                    real_stay_region = stay_label[i][j]
                    real_stay_region = real_stay_region[real_stay_region!=num_grid]
                    labeling_idx = torch.where((candidate_region_set[..., None] == real_stay_region).any(-1))[0]
                    # 存在一部分 使用candidate region 覆盖不了的驻留pair
                    where_stay_label[cnt_1][labeling_idx] = 1
                    if real_stay_region.shape[0] != 0:
                        cnt_2 += 1
                    cnt_1 += 1

        # 构造label
        where_stay_label = where_stay_label.float().cpu().detach().numpy()

        # a = torch.where(whether_stay_pred!=0)[0]
        # b = torch.where(binary_classification_label!=0)[0]
        #
        # unpadding_pair_num += cnt_1
        # stay_pair_num += cnt_2
        # candidate_region_num_list.extend(candidate_region_length)
        # detect_stay_pair_num += a.shape[0]
        # detect_real_stay_pair_num += len(set(a.numpy()).intersection(b.numpy()))

        # print(candidate_region.shape[0]*candidate_region.shape[1], cnt_1, cnt_2,
        #       round(np.mean(candidate_region_length)),
        #       len(a), len(b), len(set(a.numpy()).intersection(b.numpy())),
        #       len(set(a.numpy()).intersection(b.numpy()))/cnt_2)

        # 二分类打标为1的使用pred计算loss，打标为0的直接标位全0
        where_stay_pred[undetected_pair_idx, :, :] = 0

        where_stay_pred = where_stay_pred.squeeze(-1).cpu().detach().numpy()

        where_stay_pred_list.append(where_stay_pred[stay_pair_idx])
        where_stay_label_list.append(where_stay_label[stay_pair_idx])

        # a = np.where(where_stay_pred.reshape(-1) != 0)[0]
        # b = np.where(where_stay_label.reshape(-1) != 0)[0]
        #
        # print(where_stay_pred[where_stay_pred != 0].shape)  # 3k+
        # print(where_stay_label[where_stay_label != 0].shape) # 70+
        # print(len(list(set(a).intersection(b)))/where_stay_label[where_stay_label != 0].shape[0])

        # 对于每一个pair 看他的candidate region和stay 概率
        # for i in stay_pair_idx:
        #     print(where_stay_pred[i][0:candidate_region_length[i]])
        #     print(where_stay_label[i][0:candidate_region_length[i]])

        # hit1 = hitK(where_stay_pred[stay_pair_idx], where_stay_label[stay_pair_idx], 1)
        # hit3 = hitK(where_stay_pred[stay_pair_idx], where_stay_label[stay_pair_idx], 3)
        # hit5 = hitK(where_stay_pred[stay_pair_idx], where_stay_label[stay_pair_idx], 5)

        # 格子的概率一致表现不出区分度
        # print('acc:{:.4f}'.format(acc))
        # print('hit@1:{:.4f}'.format(hit1))
        # print('hit@3:{:.4f}'.format(hit3))
        # print('hit@5:{:.4f}'.format(hit5))
        # print('SUM:{:.4f}'.format(hit1 + hit3 + hit5))

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


    # print(unpadding_pair_num)  # 23284个pair
    # print(stay_pair_num) # 其中有3373个pair中有驻留，14.49%的pair中出现了驻留
    # print(detect_stay_pair_num) # 检测出有3632个pair有驻留，准确率是86.92%
    # print(detect_real_stay_pair_num) # 其中有3157个有驻留的pair被检测到，93.6%的pair被检测到
    # print(np.mean(candidate_region_length)) # 平均每个驻留的pair有26.7个候选集  weight 30 其实是绝对够了的

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

    exp_path = 'D:/Project/SAInf++/exp/SAInfplus_C_240121153546'
    model_name = 'SAInfplus_C_20_240121153546_19.pt'

    start_time = time.time()
    # log_path = os.path.join(exp_path, 'evaluation')
    # sys.stdout = Logger(log_path, start_time, stream=sys.stdout)  # record log
    # sys.stderr = Logger(log_path, start_time, stream=sys.stderr)  # record error
    evaluation(exp_path, model_name, test_loader, start_time)






