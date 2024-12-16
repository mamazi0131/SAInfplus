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
import shutil
from tqdm import tqdm
import argparse
from metric import hitK
from metric import stay_detection_evaluation,stay_selection_evaluation
from sklearn.metrics import accuracy_score, classification_report
from SAInf import SAInf,Detection_Threshold_Estimation


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

dev_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(dev_id)
torch.cuda.set_device(dev_id)

def train(config, train_dataloader, eval_dataloader, threshold):
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
    drop_grid_rate = config['drop_grid_rate']


    grid_feature_path = os.path.join(base_path, 'grid_feature.pkl')
    grid_feature = pickle.load(open(grid_feature_path, 'rb'))
    grid_feature = grid_standardization(grid_feature)

    grid_feature_map = {gid: feature for gid, feature in enumerate(grid_feature)}

    camera_feature_path = os.path.join(base_path, 'camera_map.pkl')
    camera_feature_map = pickle.load(open(camera_feature_path, 'rb'))
    camera_feature = [[value[0],value[1]]for value in camera_feature_map.values()]
    camera_feature = grid_standardization(camera_feature)
    

    model = SAInf(grid_feat_num,
                      num_camera, camera_embed_size,
                      num_user, user_embed_size,
                      num_grid, grid_embed_size,
                      minute_embed_size, weekofday_embed_size, weather_embed_size,
                      vehicle_type_embed_size, vehicle_color_embed_size, plate_color_embed_size,
                      hidden_dim, drop_grid_rate, threshold).cuda()

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # exp information
    nowtime = datetime.now().strftime("%y%m%d%H%M%S")
    model_name = 'SAInf_{}_{}_{}'.format(region, num_epochs, nowtime)
    model_path = os.path.join(save_path, 'SAInf_{}_{}'.format(region, nowtime), 'model')
    log_path = os.path.join(save_path, 'SAInf_{}_{}'.format(region, nowtime), 'log')

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

    grid_padding = torch.zeros(1, grid_feature.shape[1], requires_grad=False,dtype=float).cuda()
    grid_feature = torch.tensor(grid_feature,dtype=float).cuda()
    grid_feature = torch.concat([grid_padding,grid_feature],dim=0)
    model.grid_mapping.weight = nn.Parameter(grid_feature)

    camera_padding = torch.zeros(1, camera_feature.shape[1], requires_grad=False,dtype=float).cuda()
    camera_feature = torch.tensor(camera_feature,dtype=float).cuda()
    camera_feature = torch.concat([camera_padding,camera_feature],dim=0)
    model.camera_mapping.weight = nn.Parameter(camera_feature)

    epoch_step = train_loader.dataset.context_data.shape[0] // batch_size
    total_steps = epoch_step * num_epochs

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0.1 * total_steps,
                                                num_training_steps=total_steps)

    BCE_loss_2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([int(pos_weight)])).cuda()  # 调整后变好

    last_sum = 0
    for epoch in range(num_epochs):
        model.train()
        for idx, batch in enumerate(train_dataloader):
            context_data, camera_traj_data, camera_assign_mat, stay_label, \
            candidate_region, camera_pair_feature,candidate_region_feature = batch

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
            
            whether_stay_pred = model.stay_evenet_detection(camera_pair_speed).long().cpu()
            binary_classification_label = binary_classification_label.float().cpu()

            acc = accuracy_score(binary_classification_label, whether_stay_pred)

            where_stay_pred = model.where_stay_head(stay_query_pair_rep)
            where_stay_pred = torch.sigmoid(where_stay_pred)  
            where_stay_pred = where_stay_pred * pool_mask  

            detected_pair_idx = torch.where(whether_stay_pred > 0.5)[0].numpy()
            undetected_pair_idx = torch.where(whether_stay_pred <= 0.5)[0].numpy()
            stay_pair_idx = torch.where(binary_classification_label == 1)[0].numpy()


            where_stay_pred[undetected_pair_idx, :, :] = 0

            real_stay_in_where_stay_pred = where_stay_pred[stay_pair_idx].squeeze(-1) # num_pair*256*1
            real_stay_in_pool_mask = pool_mask[stay_pair_idx].squeeze(-1) # num_pair*256*1
            real_stay_in_where_stay_label = where_stay_label[stay_pair_idx] # num_pair*256

            num_candidate_region = stay_query_pair_rep.shape[1]
            rows, cols = torch.where(real_stay_in_pool_mask == 1)  
            unpadding_stay_query_idx = [row * num_candidate_region + col for row, col in zip(rows, cols)]

            where_stay_pred_prob = real_stay_in_where_stay_pred.reshape(-1)[unpadding_stay_query_idx]
            where_stay_label = real_stay_in_where_stay_label.reshape(-1)[unpadding_stay_query_idx]

            where_stay_loss = BCE_loss_2(where_stay_pred_prob, where_stay_label)

            real_stay_in_where_stay_label = real_stay_in_where_stay_label.float().cpu().detach().numpy()
            real_stay_in_where_stay_pred = real_stay_in_where_stay_pred.cpu().detach().numpy()

            hit1 = hitK(real_stay_in_where_stay_pred, real_stay_in_where_stay_label, 1)
            hit3 = hitK(real_stay_in_where_stay_pred, real_stay_in_where_stay_label, 3)
            hit5 = hitK(real_stay_in_where_stay_pred, real_stay_in_where_stay_label, 5)

            loss = where_stay_loss

            step = epoch_step * epoch + idx
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

        model.eval()
        binary_classification_label_list = []
        whether_stay_pred_list = []
        whether_stay_pred_prob_list = []
        where_stay_label_list = []
        where_stay_pred_list = []

        for idx, batch in tqdm(enumerate(eval_dataloader)):
            context_data, camera_traj_data, camera_assign_mat, stay_label, \
            candidate_region, camera_pair_feature, candidate_region_feature = batch

            context_data, camera_traj_data, camera_assign_mat, camera_pair_feature = \
                context_data.cuda(), camera_traj_data.cuda(), camera_assign_mat.cuda(), camera_pair_feature.cuda()

            _, binary_classification_label, \
            where_stay_pred, where_stay_label, pool_mask = model(camera_traj_data,
                                                                    camera_assign_mat,
                                                                    context_data,
                                                                    candidate_region,
                                                                    grid_feature_map,
                                                                    camera_pair_feature,
                                                                    candidate_region_feature,
                                                                    stay_label)
    
            whether_stay_pred = model.stay_evenet_detection(camera_pair_speed).long().cpu()
            binary_classification_label = binary_classification_label.float().cpu()

            whether_stay_pred_prob_list.append(whether_stay_pred.squeeze(-1).numpy())
            whether_stay_pred_list.append(whether_stay_pred.squeeze(-1).numpy())
            binary_classification_label_list.append(binary_classification_label)

            stay_pair_idx = torch.where(binary_classification_label==1)[0].numpy()

            real_stay_in_where_stay_pred = where_stay_pred[stay_pair_idx].squeeze(-1)  # num_pair*256*1
            real_stay_in_where_stay_label = where_stay_label[stay_pair_idx]  # num_pair*256

            real_stay_in_where_stay_pred = real_stay_in_where_stay_pred.cpu().detach().numpy()
            real_stay_in_where_stay_label = real_stay_in_where_stay_label.float().cpu().detach().numpy()

            where_stay_pred_list.append(real_stay_in_where_stay_pred)
            where_stay_label_list.append(real_stay_in_where_stay_label)
    
    
        where_stay_label = np.concatenate(where_stay_label_list)
        where_stay_pred = np.concatenate(where_stay_pred_list)

        hit1, hit3, hit5 = stay_selection_evaluation(where_stay_label, where_stay_pred)

        if hit1 + hit3 + hit5 > last_sum:
            shutil.rmtree(model_path)
            os.mkdir(model_path)
            torch.save({
                'epoch': epoch,
                'model': model,
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(model_path, "_".join([model_name, f'{epoch}.pt'])))
            last_sum = hit1 + hit3 + hit5
    return model


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

    setup_seed(seed)

    train_loader, val_loader, test_loader = get_loader(base_path, batch_size, num_worker, sequence_min_len,
                                                       sequence_max_len, candidate_threshold, num_grid,
                                                       max_candidate_grid)

    pair_speed = train_loader.dataset.camera_pair_feature[:,:,-1] # num_traj * 19 *3
    stay_label = train_loader.dataset.stay_label # num_traj * 20 * 3

    stay_distribution = [] 
    unstay_distribution = []
    for i in tqdm(range(stay_label.shape[0]), desc='estimate speed threshold'):
        for j in range(stay_label.shape[1]-1):
            speed = pair_speed[i][j] 
            if torch.isnan(speed):
                pass
            else:
                if (stay_label[i][j] != num_grid).long().sum() !=0 :
                    stay_distribution.append(round(speed.item(),4))
                else:
                    unstay_distribution.append(round(speed.item(), 4))

    Estimator = Detection_Threshold_Estimation(stay_distribution, unstay_distribution, 10000)
    threshold = Estimator.estimate_threshold()

    model = train(config, train_loader, val_loader, threshold)
