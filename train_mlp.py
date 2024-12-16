import os
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from dataloader import get_loader, grid_standardization
from utils import setup_seed, weight_init
import numpy as np
import json
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import pickle
from tqdm import tqdm
from metric import hitK
from sklearn.metrics import accuracy_score, classification_report
from components import *
from MLP import MLP

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

dev_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = str(dev_id)
torch.cuda.set_device(dev_id)

def train(config, dataloader):
    # train parm
    region = config['region']
    base_path = config['base_path']
    save_path = config['save_path']
    retrain = config['retrain']
    verbose = config['verbose']

    num_epochs = 20
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']


    # model parm
    hidden_dim = config['hidden_dim']
    num_grid = config['num_grid']

    grid_feature_path = os.path.join(base_path, 'grid_feature.pkl')
    grid_feature = pickle.load(open(grid_feature_path, 'rb'))
    grid_feature = grid_standardization(grid_feature)

    grid_feature_map = {gid: feature for gid, feature in enumerate(grid_feature)}

    model = MLP(hidden_dim).cuda()

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # exp information
    nowtime = datetime.now().strftime("%y%m%d%H%M%S")
    model_name = 'MLP_{}_{}_{}'.format(region, num_epochs, nowtime)
    model_path = os.path.join(save_path, 'MLP_{}_{}'.format(region, nowtime), 'model')
    log_path = os.path.join(save_path, 'MLP_{}_{}'.format(region, nowtime), 'log')

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

    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    
    BCE_loss_1 = nn.BCEWithLogitsLoss().cuda()  


    for epoch in range(num_epochs):
        model.train()
        for idx, batch in enumerate(dataloader):
            context_data, _, camera_assign_mat, stay_label, \
            _, camera_pair_feature,_ = batch

            context_data, camera_assign_mat, camera_pair_feature = \
                context_data.cuda(), camera_assign_mat.cuda(), camera_pair_feature.cuda()
            
            binary_classification_label_list = []
            flatten_camera_pair_feature_list = []
            for i in range(camera_assign_mat.shape[0]):
                for j in range(camera_assign_mat.shape[1] - 1):
                    if camera_assign_mat[i][j] == num_grid or camera_assign_mat[i][j + 1] == num_grid:
                        continue
                    else:
                        flatten_camera_pair_feature_list.append(camera_pair_feature[i][j].unsqueeze(0))

                    num_stay_grid = (stay_label[i][j] != num_grid).long().sum().item()
                    if num_stay_grid == 0:
                        binary_classification_label_list.append(0)
                    else:
                        binary_classification_label_list.append(1)

            camera_pair_feature = torch.cat(flatten_camera_pair_feature_list, dim=0)
            binary_classification_label = torch.tensor(binary_classification_label_list)
            
            whether_stay_pred = model(camera_pair_feature)
            
            label_class_0_idx = torch.where(binary_classification_label == 0)[0]
            label_class_1_idx = torch.where(binary_classification_label == 1)[0]
            label_class_0_idx = np.random.choice(label_class_0_idx, size=len(label_class_1_idx), replace=False, p=None)
            label_class_0_idx = torch.tensor(label_class_0_idx)
            label_idx = torch.cat([label_class_0_idx, label_class_1_idx])

            binary_classification_label = binary_classification_label.float().cuda()
            whether_stay_loss = BCE_loss_1(whether_stay_pred[label_idx],
                                           binary_classification_label[label_idx].unsqueeze(-1))

            whether_stay_pred = torch.sigmoid(whether_stay_pred)  
            binary_classification_label = binary_classification_label.float().cpu()
            whether_stay_pred = (whether_stay_pred >= 0.5).long().cpu()
            acc = accuracy_score(binary_classification_label, whether_stay_pred)

            loss = whether_stay_loss

            step = epoch_step * epoch + idx
            writer.add_scalar('metric/acc', acc, step)
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

    setup_seed(seed)

    train_loader, val_loader, test_loader = get_loader(base_path, batch_size, num_worker, sequence_min_len,
                                                       sequence_max_len, candidate_threshold, num_grid,
                                                       max_candidate_grid)

    model = train(config, train_loader)


