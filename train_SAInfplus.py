import os
from transformers import get_linear_schedule_with_warmup, AdamW
from dataloader import get_loader, grid_standardization
from utils import setup_seed, weight_init
import numpy as np
import json
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import pickle
from metric import hitK
from sklearn.metrics import accuracy_score, classification_report
from SAInfplus import SAInfplus

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

dev_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(dev_id)
torch.cuda.set_device(dev_id)


from unbalanced_loss.focal_loss import BinaryFocalLoss
from unbalanced_loss.dice_loss_nlp import BinaryDSCLoss

def train(config, dataloader):
    # train parm
    region = config['region']
    base_path = config['base_path']
    save_path = config['save_path']
    retrain = config['retrain']
    verbose = config['verbose']

    num_epochs = config['num_epochs']
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    warmup_step = config['warmup_step']
    weight_decay = config['weight_decay']

    # model parm
    grid_feat_num = config['grid_feat_num']
    num_camera = config['num_camera']
    camera_embed_size = config['camera_embed_size']
    num_user = config['num_user']
    user_embed_size = config['user_embed_size']
    num_grid = config['num_grid']
    grid_embed_size = config['grid_embed_size']
    minute_embed_size = config['minute_embed_size']
    weekofday_embed_size = config['weekofday_embed_size']
    weather_embed_size = config['weather_embed_size']
    vehicle_type_embed_size = config['vehicle_type_embed_size']
    vehicle_color_embed_size = config['vehicle_color_embed_size']
    plate_color_embed_size = config['plate_color_embed_size']
    hidden_dim = config['hidden_dim']

    pos_weight = config['pos_weight']
    drop_edge_rate = config['drop_edge_rate']
    drop_sequence_rate = config['drop_sequence_rate']
    drop_grid_rate = config['drop_grid_rate']
    max_len = config['max_len']
    edge_index_path = os.path.join(base_path, 'adj.pkl')
    edge_index = pickle.load(open(edge_index_path, 'rb'))

    grid_feature_path = os.path.join(base_path, 'grid_feature.pkl')
    grid_feature = pickle.load(open(grid_feature_path, 'rb'))
    grid_feature = grid_standardization(grid_feature)
    # 转换成字典提高候选集构造的速度
    grid_feature_map = {gid: feature for gid, feature in enumerate(grid_feature)}

    model = SAInfplus(grid_feat_num,
                      num_camera, camera_embed_size,
                      num_user, user_embed_size,
                      num_grid, grid_embed_size,
                      minute_embed_size, weekofday_embed_size, weather_embed_size,
                      vehicle_type_embed_size, vehicle_color_embed_size, plate_color_embed_size,
                      hidden_dim, edge_index, drop_edge_rate,
                      drop_sequence_rate, drop_grid_rate, max_len).cuda()

    # 定义模型  初始化参数  与定义优化器
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # exp information
    nowtime = datetime.now().strftime("%y%m%d%H%M%S")
    model_name = 'SAInfplus_{}_{}_{}'.format(region, num_epochs, nowtime)
    model_path = os.path.join(save_path, 'SAInfplus_{}_{}'.format(region, nowtime), 'model')
    log_path = os.path.join(save_path, 'SAInfplus_{}_{}'.format(region, nowtime), 'log')

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    checkpoints = [f for f in os.listdir(model_path) if f.startswith(model_name)]
    writer = SummaryWriter(log_path)
    if not retrain and checkpoints:
        checkpoint_path = os.path.join(model_path, sorted(checkpoints)[-1])
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        model.apply(weight_init)

    epoch_step = train_loader.dataset.context_data.shape[0] // batch_size
    total_steps = epoch_step * num_epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_steps,
                                                num_training_steps=total_steps)

    BCE_loss_1 = nn.BCEWithLogitsLoss().cuda()  # 负采样效果较好
    BCE_loss_2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([int(pos_weight)])).cuda()  # 调整后变好
    # BCE_loss_2 = BinaryFocalLoss()
    # BCE_loss_2 = BinaryDSCLoss()

    for epoch in range(num_epochs):
        model.train()
        for idx, batch in enumerate(dataloader):
            context_data, camera_traj_data, camera_assign_mat, stay_label, \
            candidate_region, camera_pair_feature,candidate_region_feature = batch

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

            # 负采样
            label_class_0_idx = torch.where(binary_classification_label == 0)[0]
            label_class_1_idx = torch.where(binary_classification_label == 1)[0]
            label_class_0_idx = np.random.choice(label_class_0_idx, size=len(label_class_1_idx), replace=False, p=None)
            label_class_0_idx = torch.tensor(label_class_0_idx)
            label_idx = torch.cat([label_class_0_idx, label_class_1_idx])

            # 计算loss
            binary_classification_label = binary_classification_label.float().cuda()
            whether_stay_loss = BCE_loss_1(whether_stay_pred[label_idx],
                                           binary_classification_label[label_idx].unsqueeze(-1))

            # 驻留事件检测 计算metric
            whether_stay_pred = torch.sigmoid(whether_stay_pred)  # 约束值域为0-1
            binary_classification_label = binary_classification_label.float().cpu()
            whether_stay_pred = (whether_stay_pred >= 0.5).long().cpu()
            acc = accuracy_score(binary_classification_label, whether_stay_pred)

            # 对检测为有驻留发生的pair上计算驻留概率，对检测没有驻留发生的pair直接把驻留概率置为0
            where_stay_pred = model.where_stay_head(stay_query_pair_rep)
            where_stay_pred = torch.sigmoid(where_stay_pred)  # 约束值域为0-1
            where_stay_pred = where_stay_pred * pool_mask  # padding部分置为0

            detected_pair_idx = torch.where(whether_stay_pred > 0.5)[0].numpy()
            undetected_pair_idx = torch.where(whether_stay_pred <= 0.5)[0].numpy()
            stay_pair_idx = torch.where(binary_classification_label == 1)[0].numpy()

            # 给那些漏检的pair中candidate region对应的概率都只为0
            where_stay_pred[undetected_pair_idx, :, :] = 0
            # 选出有stay的pair进行loss和metric计算
            real_stay_in_where_stay_pred = where_stay_pred[stay_pair_idx].squeeze(-1) # num_pair*256*1
            real_stay_in_pool_mask = pool_mask[stay_pair_idx].squeeze(-1) # num_pair*256*1
            real_stay_in_where_stay_label = where_stay_label[stay_pair_idx] # num_pair*256

            # 去掉padding的候选region
            num_candidate_region = stay_query_pair_rep.shape[1]
            rows, cols = torch.where(real_stay_in_pool_mask == 1)  # 标记unpadding的位置
            unpadding_stay_query_idx = [row * num_candidate_region + col for row, col in zip(rows, cols)]

            where_stay_pred_prob = real_stay_in_where_stay_pred.reshape(-1)[unpadding_stay_query_idx]
            # where_stay_pred_prob = real_stay_in_where_stay_pred.reshape(-1, 2)[unpadding_stay_query_idx]
            where_stay_label = real_stay_in_where_stay_label.reshape(-1)[unpadding_stay_query_idx]

            # candidate region set norm
            # last_row = rows[0]
            # set_cols = []
            # cnt = torch.tensor(0).cuda()
            # for row, col in zip(rows, cols):
            #     if row == last_row:
            #         set_cols.append(col)
            #     else:
            #         set_cols = torch.tensor(set_cols).cuda()
            #         if set_cols.shape[0] != 1 and where_stay_pred_prob[cnt + set_cols].std() != 0:
            #             set_cols = torch.tensor(set_cols).cuda()
            #             where_stay_pred_prob[cnt + set_cols] = where_stay_pred_prob[cnt + set_cols]/where_stay_pred_prob[cnt + set_cols].std()
            #         cnt += set_cols.shape[0]
            #         last_row = row
            #         set_cols = [col]

            # 计算loss

            where_stay_loss = BCE_loss_2(where_stay_pred_prob, where_stay_label)

            # 计算metric
            real_stay_in_where_stay_label = real_stay_in_where_stay_label.float().cpu().detach().numpy()
            real_stay_in_where_stay_pred = real_stay_in_where_stay_pred.cpu().detach().numpy()

            hit1 = hitK(real_stay_in_where_stay_pred, real_stay_in_where_stay_label, 1)
            hit3 = hitK(real_stay_in_where_stay_pred, real_stay_in_where_stay_label, 3)
            hit5 = hitK(real_stay_in_where_stay_pred, real_stay_in_where_stay_label, 5)

            # hit1 = hitK(real_stay_in_where_stay_pred[:, :, -1], real_stay_in_where_stay_label, 1)
            # hit3 = hitK(real_stay_in_where_stay_pred[:, :, -1], real_stay_in_where_stay_label, 3)
            # hit5 = hitK(real_stay_in_where_stay_pred[:, :,- 1], real_stay_in_where_stay_label, 5)

            # pair 与 stay_query 不同
            # pair 是 每两个摄像头记录构成的一个记录对
            # stay_query 是记录对间产生的候选区域

            # 驻留事件检测 与 驻留区域选择的 关系是什么
            # 1. 首先，驻留事件检测决定了哪些pair应该被关注，哪些不应该被关注
            # 2. 根据以往训练的经验来看，驻留事件检测模型会存在误判的情况，recall较高90+, presion较低85+
            # 3. 在过去的评估指标中，我们对有stay的pair进行 驻留区域选择 进行评估，成为评估1
            # 4. 在训练过程中，我们屏蔽 驻留区域选择 的loss，随着学习程度的加深，驻留事件检测的性能会变好，并且评估1的性能存在上升

            # 在计算loss的时候，我们希望使用detected pair进行处理，好处是可以保证两个loss有联动
            # 也可以使用stay pair进行处理，保证产生的损失稳定
            # 其实两种选择差别不大

            # todo 比较之前没有的特征对模型的影响

            w_1 = 2
            w_2 = 1
            loss = w_1 * whether_stay_loss + w_2 * where_stay_loss

            step = epoch_step * epoch + idx
            writer.add_scalar('item_loss/whether_stay_loss', whether_stay_loss, step)
            writer.add_scalar('item_loss/where_stay_loss', where_stay_loss, step)
            writer.add_scalar('metric/acc', acc, step)
            writer.add_scalar('metric/hit1', hit1, step)
            writer.add_scalar('metric/hit3', hit3, step)
            writer.add_scalar('metric/hit5', hit5, step)
            writer.add_scalar('loss', loss, step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not (idx + 1) % verbose:
                t = datetime.now().strftime('%m-%d %H:%M:%S')
                print(f'{t} | (Train) | Epoch={epoch}\tbatch_id={idx + 1}\tloss={loss.item():.4f}')

        scheduler.step()

        torch.save({
            'epoch': epoch,
            'model': model,
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(model_path, "_".join([model_name, f'{epoch}.pt'])))

    return model


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
                                                       sequence_max_len, candidate_threshold, num_grid,
                                                       max_candidate_grid)

    model = train(config, train_loader)


