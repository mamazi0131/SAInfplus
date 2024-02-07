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

        camera_pair_rep, binary_classification_label, \
        stay_query_pair_rep, where_stay_label, pool_mask = model(camera_traj_data,
                                                                 camera_assign_mat,
                                                                 context_data,
                                                                 candidate_region,
                                                                 grid_feature_map,
                                                                 camera_pair_feature,
                                                                 candidate_region_feature,
                                                                 stay_label)
        # 驻留事件检测
        whether_stay_pred = model.whether_stay_head(camera_pair_rep)
        whether_stay_pred = torch.sigmoid(whether_stay_pred)  # 约束值域为0-1
        whether_stay_pred_prob_list.append(whether_stay_pred.squeeze(-1).cpu().detach().numpy())

        whether_stay_pred = (whether_stay_pred >= 0.5).long().cpu()
        whether_stay_pred_list.append(whether_stay_pred.squeeze(-1))

        binary_classification_label = binary_classification_label.float().cpu()
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

    exp_path = './exp/SAInfplus_C_240207081859'
    model_name = 'SAInfplus_C_50_240207081859_39.pt'

    start_time = time.time()
    # log_path = os.path.join(exp_path, 'evaluation')
    # sys.stdout = Logger(log_path, start_time, stream=sys.stdout)  # record log
    # sys.stderr = Logger(log_path, start_time, stream=sys.stderr)  # record error
    evaluation(exp_path, model_name, test_loader, start_time,mode='detail')



# 2024 0126 第一次正式实验
# 使用默认的pos_weight进行训练

# exp_path = 'D:/Project/SAInf++/exp/SAInfplus_C_240125000224'
# model_name = 'SAInfplus_C_20_240125000224_19.pt'
# acc:0.9736 auc:0.9932
# precision:0.8754 recall:0.9538 f1:0.9129
# hit@1:0.3813 hit@3:0.5727 hit@5:0.6535 SUM:1.6075

# exp_path = 'D:/Project/SAInf++/exp/SAInfplus_A_240125042430'
# model_name = 'SAInfplus_A_20_240125042430_19.pt'
# acc:0.9716 auc:0.9939
# precision:0.8895 recall:0.9611 f1:0.9239
# hit@1:0.2704 hit@3:0.4187 hit@5:0.4804 SUM:1.1695

# exp_path = 'D:/Project/SAInf++/exp/SAInfplus_B_240125160901'
# model_name = 'SAInfplus_B_20_240125160901_19.pt'
# acc:0.9788 auc:0.9948
# precision:0.9028 recall:0.9458 f1:0.9238
# hit@1:0.2823 hit@3:0.4232 hit@5:0.4780 SUM:1.1835

# 验证发现并不是weight越大越好，而是符合数据原本的分布时较好，因此按照实际的正负样本比调整pos_weight

# 调整pos_weight 30-->44
# exp_path = 'D:/Project/SAInf++/exp/SAInfplus_C_240126154930'
# model_name = 'SAInfplus_C_20_240126154930_19.pt'
# acc:0.9708 auc:0.9932
# precision:0.8554 recall:0.9609 f1:0.9051
# hit@1:0.3718 hit@3:0.5934 hit@5:0.6816 SUM:1.6468

# 调整pos_weight 44-->100
# exp_path = 'D:/Project/SAInf++/exp/SAInfplus_C_240126205422'
# model_name = 'SAInfplus_C_20_240126205422_19.pt'
# acc:0.9692 auc:0.9919
# precision:0.8578 recall:0.9443 f1:0.8990
# hit@1:0.3101 hit@3:0.5299 hit@5:0.6416 SUM:1.4816

# 调整pos_weight 30-->55
# exp_path = 'D:/Project/SAInf++/exp/SAInfplus_A_240127130147'
# model_name = 'SAInfplus_A_20_240127130147_19.pt'
# acc:0.9701 auc:0.9937
# precision:0.8798 recall:0.9653 f1:0.9206
# hit@1:0.2787 hit@3:0.4718 hit@5:0.5772 SUM:1.3276

# 调整pos_weight 30-->80
# exp_path = 'D:/Project/SAInf++/exp/SAInfplus_B_240127151953'
# model_name = 'SAInfplus_B_20_240127151953_19.pt'
# acc:0.9788 auc:0.9949
# precision:0.9030 recall:0.9459 f1:0.9240
# hit@1:0.2757 hit@3:0.4495 hit@5:0.5457 SUM:1.2709

# 调整 w2 1 --> 3
# exp_path = 'D:/Project/SAInf++/exp/SAInfplus_C_240127222947'
# model_name = 'SAInfplus_C_20_240127222947_19.pt'
# acc:0.9722 auc:0.9919
# precision:0.8999 recall:0.9090 f1:0.9044
# hit@1:0.3698 hit@3:0.5768 hit@5:0.6694 SUM:1.6161

# 调整 w1 1-->0
# exp_path = 'D:/Project/SAInf++/exp/SAInfplus_C_240128021215'
# model_name = 'SAInfplus_C_20_240128021215_19.pt'
# acc:0.9768 auc:0.9933
# precision:0.9011 recall:0.9431 f1:0.9216
# hit@1:0.2447 hit@3:0.3695 hit@5:0.4459 SUM:1.0600

# 使用focal loss
# exp_path = 'D:/Project/SAInf++/exp/SAInfplus_C_240128040205'
# model_name = 'SAInfplus_C_20_240128040205_19.pt'
# acc:0.9747 auc:0.9929
# precision:0.8897 recall:0.9422 f1:0.9152
# hit@1:0.2344 hit@3:0.3593 hit@5:0.4379 SUM:1.0316

# 使用set norm
# exp_path = 'D:/Project/SAInf++/exp/SAInfplus_C_240128062144'
# model_name = 'SAInfplus_C_20_240128062144_19.pt'
# acc:0.9636 auc:0.9944
# precision:0.8074 recall:0.9834 f1:0.8868
# hit@1:0.2466 hit@3:0.3973 hit@5:0.4813 SUM:1.1252

# 使用dice loss
# 结果未跑完，训练时loss不动，认为存在训练问题，故放弃

# 改变attention的嵌入维度 256-->4
# 结果未跑完，训练时曲线与不改动几乎无差别，故放弃

# 改变head维度 32-->64
# 结果未跑完，训练时曲线观察到模型性能略微提升，提升较小

# 改变head all mlp-->linear
# detection任务下降明显，但是ranker任务有提升
# exp_path = 'D:/Project/SAInf++/exp/SAInfplus_C_240129040940'
# model_name = 'SAInfplus_C_20_240129040940_19.pt'
# acc:0.9661 auc:0.9880
# precision:0.8915 recall:0.8719 f1:0.8816
# hit@1:0.3714 hit@3:0.5806 hit@5:0.6678 SUM:1.6199

# 改变head detection mlp-->linear
# exp_path = 'D:/Project/SAInf++/exp/SAInfplus_C_240129094448'
# model_name = 'SAInfplus_C_20_240129094448_19.pt'
# acc:0.9755 auc:0.9931
# precision:0.8935 recall:0.9431 f1:0.9176
# hit@1:0.3705 hit@3:0.6027 hit@5:0.7023 SUM:1.6755

# 改变head detection mlp-->linear
# exp_path = 'D:/Project/SAInf++/exp/SAInfplus_A_240129121220'
# model_name = 'SAInfplus_A_20_240129121220_19.pt'
# acc:0.9738 auc:0.9945
# precision:0.8982 recall:0.9634 f1:0.9297
# hit@1:0.2846 hit@3:0.4869 hit@5:0.5907 SUM:1.3623

# 改变head detection mlp-->linear
# exp_path = 'D:/Project/SAInf++/exp/SAInfplus_B_240129150632'
# model_name = 'SAInfplus_B_20_240129150632_19.pt'
# acc:0.9791 auc:0.9950
# precision:0.8975 recall:0.9553 f1:0.9255
# hit@1:0.2669 hit@3:0.4513 hit@5:0.5469 SUM:1.2651

# 多头注意力
# exp_path = 'D:/Project/SAInf++/exp/SAInfplus_C_240130040027'
# model_name = 'SAInfplus_C_20_240130040027_19.pt'
# acc:0.9768 auc:0.9935
# precision:0.9046 recall:0.9389 f1:0.9214
# hit@1:0.3804 hit@3:0.5909 hit@5:0.6918 SUM:1.6630

# candidate region encoder的head数量减少
# exp_path = 'D:/Project/SAInf++/exp/SAInfplus_C_240130124953'
# model_name = 'SAInfplus_C_20_240130124953_19.pt'
# acc:0.9768 auc:0.9933
# precision:0.9060 recall:0.9371 f1:0.9213
# hit@1:0.3801 hit@3:0.6004 hit@5:0.6972 SUM:1.6777

# 展开详细分析

# 变差了，小心
# acc:0.9689 auc:0.9926
# precision:0.9446 recall:0.8340 f1:0.8858
# hit@1:0.3727 hit@3:0.5759 hit@5:0.6723 SUM:1.6209

# acc:0.9787 auc:0.9949
# precision:0.9200 recall:0.9339 f1:0.9269
# hit@1:0.3762 hit@3:0.5730 hit@5:0.6710 SUM:1.6202

# exp_path = 'D:/Project/SAInf++/exp/SAInfplus_C_240205172932'
# model_name = 'SAInfplus_C_20_240205172932_19.pt'
# acc:0.9741 auc:0.9933
# precision:0.9280 recall:0.8900 f1:0.9086
# hit@1:0.3960 hit@3:0.6675 hit@5:0.7790 SUM:1.8425

# exp_path = 'D:/Project/SAInf++/exp/SAInfplus_A_240205191235'
# model_name = 'SAInfplus_A_20_240205191235_19.pt'
# acc:0.9751 auc:0.9946
# precision:0.9081 recall:0.9582 f1:0.9325
# hit@1:0.2905 hit@3:0.5493 hit@5:0.7018 SUM:1.5416


# exp_path = 'D:/Project/SAInf++/exp/SAInfplus_B_240205220644'
# model_name = 'SAInfplus_B_20_240205220644_19.pt'
# acc:0.9791 auc:0.9951
# precision:0.9024 recall:0.9492 f1:0.9252
# hit@1:0.3000 hit@3:0.5275 hit@5:0.6539 SUM:1.4814

# no embedding
# exp_path = 'D:/Project/SAInf++/exp/SAInfplus_C_240206140228'
# model_name = 'SAInfplus_C_20_240206140228_19.pt'
# acc:0.9699 auc:0.9922
# precision:0.9290 recall:0.8574 f1:0.8918
# hit@1:0.4069 hit@3:0.6682 hit@5:0.7729 SUM:1.8480

# no embedding whether loss 2
# exp_path = 'D:/Project/SAInf++/exp/SAInfplus_C_240206140228'
# model_name = 'SAInfplus_C_20_240206140228_19.pt'
# acc:0.9723 auc:0.9926
# precision:0.9228 recall:0.8829 f1:0.9024
# hit@1:0.4082 hit@3:0.6669 hit@5:0.7812 SUM:1.8563

# embedding 做的越好 hit@1 越高 但 hit@3和hit@5的性能越差

# 不使用车辆车身车辆类型 embedding
# exp_path = './exp/SAInfplus_A_240206183034'
# model_name = 'SAInfplus_A_20_240206183034_19.pt'
# acc:0.9715 auc:0.9937
# precision:0.8906 recall:0.9591 f1:0.9236
# hit@1:0.2904 hit@3:0.5578 hit@5:0.7057 SUM:1.5539

# exp_path = 'D:/Project/SAInf++/exp/SAInfplus_B_240206185056'
# model_name = 'SAInfplus_B_20_240206185056_19.pt'
# acc:0.9778 auc:0.9946
# precision:0.8923 recall:0.9514 f1:0.9209
# hit@1:0.3024 hit@3:0.5247 hit@5:0.6517 SUM:1.4788


# exp_path = './exp/SAInfplus_C_240206235632'
# model_name = 'SAInfplus_C_50_240206235632_49.pt'
# acc:0.9767 auc:0.9941
# precision:0.9078 recall:0.9342 f1:0.9208
# hit@1:0.4098 hit@3:0.6825 hit@5:0.7959 SUM:1.8882

# exp_path = './exp/SAInfplus_C_240206235632'
# model_name = 'SAInfplus_C_50_240206235632_49.pt'
# acc:0.9767 auc:0.9941
# precision:0.9078 recall:0.9342 f1:0.9208
# hit@1:0.4098 hit@3:0.6825 hit@5:0.7959 SUM:1.8882

# 2 倍 where loss
# exp_path = 'D:/Project/SAInf++/exp/SAInfplus_C_240207035523'
# model_name = 'SAInfplus_C_20_240207035523_19.pt'
# acc:0.9710 auc:0.9925
# precision:0.9290 recall:0.8657 f1:0.8963
# hit@1:0.4037 hit@3:0.6627 hit@5:0.7723 SUM:1.8387

# 2 倍 whether loss 40 epoch
# exp_path = './exp/SAInfplus_C_240207081859'
# model_name = 'SAInfplus_C_50_240207081859_39.pt'
# acc:0.9732 auc:0.9921
# precision:0.9039 recall:0.9117 f1:0.9077
# hit@1:0.4158 hit@3:0.6742 hit@5:0.7799 SUM:1.8700



# 40 epoch no embedding loss weight=1:1





