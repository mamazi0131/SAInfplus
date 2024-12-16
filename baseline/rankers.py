import torch
import torch.nn as nn
from basemodel import BaseModel
import sys
sys.path.append('./')
from components import *
import numpy as np
from scipy.interpolate import interp1d

# 需要注意，这些方法的输入输出是
# 输入[query,region] 经过mlp/rnn/transformer 直接输出 p
# 
# mlp           point-wise encoding point-wise query
# rnn           set-wise encoding，point-wise query  
# transformer   set-wise encoding，point-wise query

class MLP(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2):
        super(MLP, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.ReLU(True)) 
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, 1))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x) # batchsize*max_len
        return x

class RNN(BaseModel):
    def __init__(self, in_dim, hidden_dim):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(in_dim, hidden_dim, 2, batch_first=True)

    def forward(self, x, mask):
        x_hidden = None
        x_out, _ = self.rnn(x, x_hidden)
        if mask is not None:
            mask = mask.unsqueeze(-1).expand_as(x_out).byte()
            x_out = x_out.masked_fill(mask, 0)
        return x_out


# 需要注意 pair encoder是SAInf设计的结构，在baseline中不存在
class SAInf_rankmlp(BaseModel):
    def __init__(self, num_camera, num_grid, hidden_dim, threshold):
        super(SAInf_rankmlp, self).__init__()

        self.num_grid = num_grid
        self.hidden_dim = hidden_dim
        self.threshold = threshold

        # 编码格子 使用POI 可以和目前构造的23维特征公用
        # grid 获取特征 23 维特征
        self.grid_mapping = nn.Embedding(num_grid+1, 23) # 0是padding
        self.grid_mapping.requires_grad_(False)

        # camera 获取特征 经纬度
        self.camera_mapping = nn.Embedding(num_camera+1, 2) # 0是padding
        self.camera_mapping.requires_grad_(False)
        
        # weather
        self.weather_embedding = nn.Embedding(10, 2)
        self.weather_embedding.requires_grad_(True)

        # minute embedding
        self.minute_embedding = nn.Embedding(1440+1, 2) # 0是padding
        self.minute_embedding.requires_grad_(True)

        # weekofday embedding
        self.weekofday_embedding = nn.Embedding(7, 2)
        self.weekofday_embedding.requires_grad_(True)

        camera_record_feat_num = 2 + 23 + 2  
        
        # context transformation
        # week, weather, temperature
        context_feat_num = 2 + 2 + 1
        pair_feat_num = 3
        in_dim = context_feat_num + pair_feat_num + 3*camera_record_feat_num

        self.where_stay_head = MLP(in_dim,hidden_dim,64)

    def stay_evenet_detection(self, camera_pair_speed):
        # camera_pair_speed 维度为1 指两个pair间的speed
        return (camera_pair_speed <= self.threshold).long()

    def prepare_context(self, context_data):
        # usr_id, week, weather, temperature
        # size * max_len * 4

        weekofday_emb = self.weekofday_embedding(context_data[:, 1].long())
        weather_emb = self.weather_embedding(context_data[:, 2].long())
        temperature_value = context_data[:, 3].unsqueeze(-1)

        context_feat = torch.cat([weekofday_emb, weather_emb, temperature_value], dim=-1)
        
        return context_feat

    def prepare_camera_pair(self, camera_traj_data, context_data, camera_assign_mat, camera_pair_feature, stay_label):
        # cllx，cpys，csys，minute，timestamp，cid, gid

        camera_traj_data = camera_traj_data + 1 # padding位置从-1变为0，统一embedding的位置一样，方便后面mask

        minute_emb = self.minute_embedding(camera_traj_data[:, :, 3].long())

        # camera_traj_data[:, :, 5].long() --> 经纬度
        # camera_traj_data[:, :, 6].long() --> grid feature
        camera_feature = self.camera_mapping(camera_traj_data[:, :, 5].long())
        grid_feature = self.grid_mapping(camera_traj_data[:, :, 6].long())
        # 两个hard mapping 通过 static embeding实现

        camera_traj_feature = torch.cat([minute_emb, camera_feature, grid_feature], dim=-1) # batch_size * max_len * (64*4)
        
        context_feature = self.prepare_context(context_data)

        # todo 先单独过linear layer再拼接
        # todo 先拼接，再一起过linear layer

        camera_pair_feature_list = []
        binary_classification_label_list = []
        camera_pair_speed_list = []
        for i in range(camera_assign_mat.shape[0]):
            for j in range(camera_assign_mat.shape[1] - 1):
                if camera_assign_mat[i][j] == self.num_grid or camera_assign_mat[i][j + 1] == self.num_grid:
                    continue
                else:
                    # camera_assign_mat中第 j 和 j+1 的记录对 对应的是camera_pair_feature 中的第j个
                    pair_fea = torch.cat( \
                        [context_feature[i], camera_pair_feature[i][j], camera_traj_feature[i][j], camera_traj_feature[i][j + 1]], dim=-1)
                    camera_pair_feature_list.append(pair_fea.unsqueeze(0))

                    camera_pair_speed = camera_pair_feature[i][j][-1]
                    camera_pair_speed_list.append(camera_pair_speed)

                num_stay_grid = (stay_label[i][j] != self.num_grid).long().sum().item()
                if num_stay_grid == 0:
                    binary_classification_label_list.append(0)
                else:
                    binary_classification_label_list.append(1)

        camera_pair_fea = torch.cat(camera_pair_feature_list, dim=0)
        binary_classification_label = torch.tensor(binary_classification_label_list)
        camera_pair_speed = torch.tensor(camera_pair_speed_list)
        return camera_pair_fea, binary_classification_label, camera_pair_speed

    def recall_candidate_grid(self, candidate_region, grid_feature_map, candidate_region_feature, stay_label):
        # 将以traj为单位的组织方式转化成以pair为单位的组织方式，traj中的padding处去除
        # 对于候选集已经预先选定好的，应该放到 dataloader 中处理 这样计算效率高一些
        # num_traj * (max_len-1) * max_stay_grid
        # 64*19*256

        num_feature = len(list(grid_feature_map.values())[0])
        pair_candidate_region_list = []

        region_feature_list = []
        where_stay_label_list = []
        for i in range(candidate_region.shape[0]):  # -1为padding位
            for j in range(candidate_region.shape[1]):
                candidate_region_set = candidate_region[i][j]
                candidate_region_set_feature = candidate_region_feature[i][j]
                if (
                        candidate_region_set != self.num_grid).long().sum().item() == 0:  # 在构造candidate_region时，padding的位置的候选集全部都为num_grid
                    continue  # padding pair位置pass
                else:  # 非padding pair
                    pair_candidate_region_list.append(candidate_region_set.unsqueeze(0))
                    this_region_feature_list = []
                    for grid in candidate_region_set:
                        try:
                            this_region_feature_list.append(
                                grid_feature_map[int(grid)])  # candidate_region 在cpu上，int只会取值不会发生设备转移
                        except:
                            this_region_feature_list.append([float('nan')] * num_feature)

                    this_region_feature = torch.tensor(this_region_feature_list, dtype=torch.float)  # max_stay_grid*23
                    # this_region_feature = torch.cat([this_region_feature, candidate_region_set_feature], dim=-1)
                    this_region_feature = candidate_region_set_feature

                    region_feature_list.append(this_region_feature.unsqueeze(0))

                    candidate_region_set = candidate_region_set[candidate_region_set != self.num_grid]

                    stay_region = stay_label[i][j]
                    stay_region = stay_region[stay_region != self.num_grid]

                    label_idx = torch.where((candidate_region_set[..., None] == stay_region).any(-1))[0]
                    label = torch.zeros(candidate_region.shape[-1])
                    label[label_idx] = 1

                    where_stay_label_list.append(label.unsqueeze(0))

        region_feature = torch.cat(region_feature_list, dim=0).cuda()  # num_pair * max_stay_grid * 23
        where_stay_label = torch.cat(where_stay_label_list, dim=0).cuda()
        pair_candidate_region = torch.cat(pair_candidate_region_list, dim=0).cuda()

        return region_feature, pair_candidate_region, where_stay_label

    def prepare_canidate_grid(self, region_feature, pair_candidate_region):

        src_key_padding_mask = torch.isnan(region_feature)[:, :, 0]
        pool_mask = (1 - src_key_padding_mask.int()).unsqueeze(-1)  # 0 为padding位

        region_feature_1 = torch.nan_to_num(region_feature, nan=0.0) # norm之前，否则mean和std会计算为0，使linear参数归零
        region_feature_2 = self.grid_mapping(pair_candidate_region)

        region_feature = torch.cat([region_feature_1, region_feature_2], dim=-1)

        region_feature = region_feature * pool_mask.repeat(1, 1, region_feature.shape[
            -1])
        return region_feature, pool_mask

    def prepare_query(self, camera_pair_feature, region_feature, pool_mask):

        camera_pair_feature = camera_pair_feature.unsqueeze(1).repeat(1,region_feature.shape[1],1)

        stay_query_pair_feature = torch.cat([camera_pair_feature, region_feature], dim=-1)
        stay_query_pair_feature = stay_query_pair_feature * pool_mask.repeat(1, 1, stay_query_pair_feature.shape[-1])
        return stay_query_pair_feature

    def forward(self, camera_traj_data, camera_assign_mat, context_data, candidate_region, \
                grid_feature_map, camera_pair_feature, candidate_region_feature, stay_label):
        camera_pair_fea, binary_classification_label, camera_pair_speed = self.prepare_camera_pair(camera_traj_data, context_data, camera_assign_mat, camera_pair_feature, stay_label)
        region_feature, pair_candidate_region, where_stay_label = self.recall_candidate_grid(candidate_region, grid_feature_map,
                                                                      candidate_region_feature,
                                                                      stay_label)
        region_feature, pool_mask = self.prepare_canidate_grid(region_feature, pair_candidate_region)
                
        stay_query_pair_feature = self.prepare_query(camera_pair_fea, region_feature, pool_mask)
        stay_query_pair_feature = stay_query_pair_feature.to(torch.float)
        
        return camera_pair_speed, binary_classification_label, stay_query_pair_feature, where_stay_label, pool_mask

class SAInf_rankrnn(BaseModel):
    def __init__(self, num_camera, num_grid, hidden_dim, threshold):
        super(SAInf_rankrnn, self).__init__()

        self.num_grid = num_grid
        self.hidden_dim = hidden_dim
        self.threshold = threshold

        # 编码格子 使用POI 可以和目前构造的23维特征公用
        # grid 获取特征 23 维特征
        self.grid_mapping = nn.Embedding(num_grid+1, 23) # 0是padding
        self.grid_mapping.requires_grad_(False)

        # camera 获取特征 经纬度
        self.camera_mapping = nn.Embedding(num_camera+1, 2) # 0是padding
        self.camera_mapping.requires_grad_(False)
        
        # weather
        self.weather_embedding = nn.Embedding(10, 2)
        self.weather_embedding.requires_grad_(True)

        # minute embedding
        self.minute_embedding = nn.Embedding(1440+1, 2) # 0是padding
        self.minute_embedding.requires_grad_(True)

        # weekofday embedding
        self.weekofday_embedding = nn.Embedding(7, 2)
        self.weekofday_embedding.requires_grad_(True)

        camera_record_feat_num = 2 + 23 + 2  
        
        # context transformation
        # week, weather, temperature
        context_feat_num = 2 + 2 + 1
        pair_feat_num = 3
        in_dim = context_feat_num + pair_feat_num + 3*camera_record_feat_num

        self.where_stay_head = nn.Sequential(nn.Linear(hidden_dim, 1))

        self.set_wise_encoder = RNN(in_dim,hidden_dim)

    def stay_evenet_detection(self, camera_pair_speed):
        # camera_pair_speed 维度为1 指两个pair间的speed
        return (camera_pair_speed <= self.threshold).long()

    def prepare_context(self, context_data):
        # usr_id, week, weather, temperature
        # size * max_len * 4

        weekofday_emb = self.weekofday_embedding(context_data[:, 1].long())
        weather_emb = self.weather_embedding(context_data[:, 2].long())
        temperature_value = context_data[:, 3].unsqueeze(-1)

        context_feat = torch.cat([weekofday_emb, weather_emb, temperature_value], dim=-1)
        
        return context_feat

    def prepare_camera_pair(self, camera_traj_data, context_data, camera_assign_mat, camera_pair_feature, stay_label):
        # cllx，cpys，csys，minute，timestamp，cid, gid

        camera_traj_data = camera_traj_data + 1 # padding位置从-1变为0，统一embedding的位置一样，方便后面mask

        minute_emb = self.minute_embedding(camera_traj_data[:, :, 3].long())

        # camera_traj_data[:, :, 5].long() --> 经纬度
        # camera_traj_data[:, :, 6].long() --> grid feature
        camera_feature = self.camera_mapping(camera_traj_data[:, :, 5].long())
        grid_feature = self.grid_mapping(camera_traj_data[:, :, 6].long())
        # 两个hard mapping 通过 static embeding实现

        camera_traj_feature = torch.cat([minute_emb, camera_feature, grid_feature], dim=-1) # batch_size * max_len * (64*4)
        
        context_feature = self.prepare_context(context_data)

        # todo 先单独过linear layer再拼接
        # todo 先拼接，再一起过linear layer

        camera_pair_feature_list = []
        binary_classification_label_list = []
        camera_pair_speed_list = []
        for i in range(camera_assign_mat.shape[0]):
            for j in range(camera_assign_mat.shape[1] - 1):
                if camera_assign_mat[i][j] == self.num_grid or camera_assign_mat[i][j + 1] == self.num_grid:
                    continue
                else:
                    # camera_assign_mat中第 j 和 j+1 的记录对 对应的是camera_pair_feature 中的第j个
                    pair_fea = torch.cat( \
                        [context_feature[i], camera_pair_feature[i][j], camera_traj_feature[i][j], camera_traj_feature[i][j + 1]], dim=-1)
                    camera_pair_feature_list.append(pair_fea.unsqueeze(0))

                    camera_pair_speed = camera_pair_feature[i][j][-1]
                    camera_pair_speed_list.append(camera_pair_speed)

                num_stay_grid = (stay_label[i][j] != self.num_grid).long().sum().item()
                if num_stay_grid == 0:
                    binary_classification_label_list.append(0)
                else:
                    binary_classification_label_list.append(1)

        camera_pair_fea = torch.cat(camera_pair_feature_list, dim=0)
        binary_classification_label = torch.tensor(binary_classification_label_list)
        camera_pair_speed = torch.tensor(camera_pair_speed_list)
        return camera_pair_fea, binary_classification_label, camera_pair_speed

    def recall_candidate_grid(self, candidate_region, grid_feature_map, candidate_region_feature, stay_label):
        # 将以traj为单位的组织方式转化成以pair为单位的组织方式，traj中的padding处去除
        # 对于候选集已经预先选定好的，应该放到 dataloader 中处理 这样计算效率高一些
        # num_traj * (max_len-1) * max_stay_grid
        # 64*19*256

        num_feature = len(list(grid_feature_map.values())[0])
        pair_candidate_region_list = []

        region_feature_list = []
        where_stay_label_list = []
        for i in range(candidate_region.shape[0]):  # -1为padding位
            for j in range(candidate_region.shape[1]):
                candidate_region_set = candidate_region[i][j]
                candidate_region_set_feature = candidate_region_feature[i][j]
                if (
                        candidate_region_set != self.num_grid).long().sum().item() == 0:  # 在构造candidate_region时，padding的位置的候选集全部都为num_grid
                    continue  # padding pair位置pass
                else:  # 非padding pair
                    pair_candidate_region_list.append(candidate_region_set.unsqueeze(0))
                    this_region_feature_list = []
                    for grid in candidate_region_set:
                        try:
                            this_region_feature_list.append(
                                grid_feature_map[int(grid)])  # candidate_region 在cpu上，int只会取值不会发生设备转移
                        except:
                            this_region_feature_list.append([float('nan')] * num_feature)

                    this_region_feature = torch.tensor(this_region_feature_list, dtype=torch.float)  # max_stay_grid*23
                    # this_region_feature = torch.cat([this_region_feature, candidate_region_set_feature], dim=-1)
                    this_region_feature = candidate_region_set_feature

                    region_feature_list.append(this_region_feature.unsqueeze(0))

                    candidate_region_set = candidate_region_set[candidate_region_set != self.num_grid]

                    stay_region = stay_label[i][j]
                    stay_region = stay_region[stay_region != self.num_grid]

                    label_idx = torch.where((candidate_region_set[..., None] == stay_region).any(-1))[0]
                    label = torch.zeros(candidate_region.shape[-1])
                    label[label_idx] = 1

                    where_stay_label_list.append(label.unsqueeze(0))

        region_feature = torch.cat(region_feature_list, dim=0).cuda()  # num_pair * max_stay_grid * 23
        where_stay_label = torch.cat(where_stay_label_list, dim=0).cuda()
        pair_candidate_region = torch.cat(pair_candidate_region_list, dim=0).cuda()

        return region_feature, pair_candidate_region, where_stay_label

    def prepare_canidate_grid(self, region_feature, pair_candidate_region):

        src_key_padding_mask = torch.isnan(region_feature)[:, :, 0]
        pool_mask = (1 - src_key_padding_mask.int()).unsqueeze(-1)  # 0 为padding位

        region_feature_1 = torch.nan_to_num(region_feature, nan=0.0) # norm之前，否则mean和std会计算为0，使linear参数归零
        region_feature_2 = self.grid_mapping(pair_candidate_region)

        region_feature = torch.cat([region_feature_1, region_feature_2], dim=-1)

        region_feature = region_feature * pool_mask.repeat(1, 1, region_feature.shape[
            -1])
        return region_feature, pool_mask, src_key_padding_mask

    def prepare_query(self, camera_pair_feature, region_feature, pool_mask):

        camera_pair_feature = camera_pair_feature.unsqueeze(1).repeat(1,region_feature.shape[1],1)

        stay_query_pair_feature = torch.cat([camera_pair_feature, region_feature], dim=-1)
        stay_query_pair_feature = stay_query_pair_feature * pool_mask.repeat(1, 1, stay_query_pair_feature.shape[-1])
        return stay_query_pair_feature

    def forward(self, camera_traj_data, camera_assign_mat, context_data, candidate_region, \
                grid_feature_map, camera_pair_feature, candidate_region_feature, stay_label):
        camera_pair_fea, binary_classification_label, camera_pair_speed = self.prepare_camera_pair(camera_traj_data, context_data, camera_assign_mat, camera_pair_feature, stay_label)
        region_feature, pair_candidate_region, where_stay_label = self.recall_candidate_grid(candidate_region, grid_feature_map,
                                                                      candidate_region_feature,
                                                                      stay_label)
        region_feature, pool_mask, src_key_padding_mask = self.prepare_canidate_grid(region_feature, pair_candidate_region)
                
        stay_query_pair_feature = self.prepare_query(camera_pair_fea, region_feature, pool_mask)
        stay_query_pair_feature = stay_query_pair_feature.to(torch.float)
        
        stay_query_pair_rep = self.set_wise_encoder(stay_query_pair_feature, src_key_padding_mask)
        return camera_pair_speed, binary_classification_label, stay_query_pair_rep, where_stay_label, pool_mask

class SAInf_ranktransformer(BaseModel):
    def __init__(self, num_camera, num_grid, hidden_dim, threshold):
        super(SAInf_ranktransformer, self).__init__()

        self.num_grid = num_grid
        self.hidden_dim = hidden_dim
        self.threshold = threshold

        # 编码格子 使用POI 可以和目前构造的23维特征公用
        # grid 获取特征 23 维特征
        self.grid_mapping = nn.Embedding(num_grid+1, 23) # 0是padding
        self.grid_mapping.requires_grad_(False)

        # camera 获取特征 经纬度
        self.camera_mapping = nn.Embedding(num_camera+1, 2) # 0是padding
        self.camera_mapping.requires_grad_(False)
        
        # weather
        self.weather_embedding = nn.Embedding(10, 2)
        self.weather_embedding.requires_grad_(True)

        # minute embedding
        self.minute_embedding = nn.Embedding(1440+1, 2) # 0是padding
        self.minute_embedding.requires_grad_(True)

        # weekofday embedding
        self.weekofday_embedding = nn.Embedding(7, 2)
        self.weekofday_embedding.requires_grad_(True)

        camera_record_feat_num = 2 + 23 + 2  
        
        # context transformation
        # week, weather, temperature
        context_feat_num = 2 + 2 + 1
        pair_feat_num = 3
        in_dim = context_feat_num + pair_feat_num + 3*camera_record_feat_num

        self.where_stay_head = nn.Sequential(nn.Linear(hidden_dim, 1))

        self.linear = nn.Linear(in_dim, hidden_dim)
        self.set_wise_encoder = TransformerModel(hidden_dim, 8, hidden_dim, 2, 0.1)

    def stay_evenet_detection(self, camera_pair_speed):
        # camera_pair_speed 维度为1 指两个pair间的speed
        return (camera_pair_speed <= self.threshold).long()

    def prepare_context(self, context_data):
        # usr_id, week, weather, temperature
        # size * max_len * 4

        weekofday_emb = self.weekofday_embedding(context_data[:, 1].long())
        weather_emb = self.weather_embedding(context_data[:, 2].long())
        temperature_value = context_data[:, 3].unsqueeze(-1)

        context_feat = torch.cat([weekofday_emb, weather_emb, temperature_value], dim=-1)
        
        return context_feat

    def prepare_camera_pair(self, camera_traj_data, context_data, camera_assign_mat, camera_pair_feature, stay_label):
        # cllx，cpys，csys，minute，timestamp，cid, gid

        camera_traj_data = camera_traj_data + 1 # padding位置从-1变为0，统一embedding的位置一样，方便后面mask

        minute_emb = self.minute_embedding(camera_traj_data[:, :, 3].long())

        # camera_traj_data[:, :, 5].long() --> 经纬度
        # camera_traj_data[:, :, 6].long() --> grid feature
        camera_feature = self.camera_mapping(camera_traj_data[:, :, 5].long())
        grid_feature = self.grid_mapping(camera_traj_data[:, :, 6].long())
        # 两个hard mapping 通过 static embeding实现

        camera_traj_feature = torch.cat([minute_emb, camera_feature, grid_feature], dim=-1) # batch_size * max_len * (64*4)
        
        context_feature = self.prepare_context(context_data)

        # todo 先单独过linear layer再拼接
        # todo 先拼接，再一起过linear layer

        camera_pair_feature_list = []
        binary_classification_label_list = []
        camera_pair_speed_list = []
        for i in range(camera_assign_mat.shape[0]):
            for j in range(camera_assign_mat.shape[1] - 1):
                if camera_assign_mat[i][j] == self.num_grid or camera_assign_mat[i][j + 1] == self.num_grid:
                    continue
                else:
                    # camera_assign_mat中第 j 和 j+1 的记录对 对应的是camera_pair_feature 中的第j个
                    pair_fea = torch.cat( \
                        [context_feature[i], camera_pair_feature[i][j], camera_traj_feature[i][j], camera_traj_feature[i][j + 1]], dim=-1)
                    camera_pair_feature_list.append(pair_fea.unsqueeze(0))

                    camera_pair_speed = camera_pair_feature[i][j][-1]
                    camera_pair_speed_list.append(camera_pair_speed)

                num_stay_grid = (stay_label[i][j] != self.num_grid).long().sum().item()
                if num_stay_grid == 0:
                    binary_classification_label_list.append(0)
                else:
                    binary_classification_label_list.append(1)

        camera_pair_fea = torch.cat(camera_pair_feature_list, dim=0)
        binary_classification_label = torch.tensor(binary_classification_label_list)
        camera_pair_speed = torch.tensor(camera_pair_speed_list)
        return camera_pair_fea, binary_classification_label, camera_pair_speed

    def recall_candidate_grid(self, candidate_region, grid_feature_map, candidate_region_feature, stay_label):
        # 将以traj为单位的组织方式转化成以pair为单位的组织方式，traj中的padding处去除
        # 对于候选集已经预先选定好的，应该放到 dataloader 中处理 这样计算效率高一些
        # num_traj * (max_len-1) * max_stay_grid
        # 64*19*256

        num_feature = len(list(grid_feature_map.values())[0])
        pair_candidate_region_list = []

        region_feature_list = []
        where_stay_label_list = []
        for i in range(candidate_region.shape[0]):  # -1为padding位
            for j in range(candidate_region.shape[1]):
                candidate_region_set = candidate_region[i][j]
                candidate_region_set_feature = candidate_region_feature[i][j]
                if (
                        candidate_region_set != self.num_grid).long().sum().item() == 0:  # 在构造candidate_region时，padding的位置的候选集全部都为num_grid
                    continue  # padding pair位置pass
                else:  # 非padding pair
                    pair_candidate_region_list.append(candidate_region_set.unsqueeze(0))
                    this_region_feature_list = []
                    for grid in candidate_region_set:
                        try:
                            this_region_feature_list.append(
                                grid_feature_map[int(grid)])  # candidate_region 在cpu上，int只会取值不会发生设备转移
                        except:
                            this_region_feature_list.append([float('nan')] * num_feature)

                    this_region_feature = torch.tensor(this_region_feature_list, dtype=torch.float)  # max_stay_grid*23
                    # this_region_feature = torch.cat([this_region_feature, candidate_region_set_feature], dim=-1)
                    this_region_feature = candidate_region_set_feature

                    region_feature_list.append(this_region_feature.unsqueeze(0))

                    candidate_region_set = candidate_region_set[candidate_region_set != self.num_grid]

                    stay_region = stay_label[i][j]
                    stay_region = stay_region[stay_region != self.num_grid]

                    label_idx = torch.where((candidate_region_set[..., None] == stay_region).any(-1))[0]
                    label = torch.zeros(candidate_region.shape[-1])
                    label[label_idx] = 1

                    where_stay_label_list.append(label.unsqueeze(0))

        region_feature = torch.cat(region_feature_list, dim=0).cuda()  # num_pair * max_stay_grid * 23
        where_stay_label = torch.cat(where_stay_label_list, dim=0).cuda()
        pair_candidate_region = torch.cat(pair_candidate_region_list, dim=0).cuda()

        return region_feature, pair_candidate_region, where_stay_label

    def prepare_canidate_grid(self, region_feature, pair_candidate_region):

        src_key_padding_mask = torch.isnan(region_feature)[:, :, 0]
        pool_mask = (1 - src_key_padding_mask.int()).unsqueeze(-1)  # 0 为padding位

        region_feature_1 = torch.nan_to_num(region_feature, nan=0.0) # norm之前，否则mean和std会计算为0，使linear参数归零
        region_feature_2 = self.grid_mapping(pair_candidate_region)

        region_feature = torch.cat([region_feature_1, region_feature_2], dim=-1)

        region_feature = region_feature * pool_mask.repeat(1, 1, region_feature.shape[
            -1])
        return region_feature, pool_mask, src_key_padding_mask

    def prepare_query(self, camera_pair_feature, region_feature, pool_mask):

        camera_pair_feature = camera_pair_feature.unsqueeze(1).repeat(1,region_feature.shape[1],1)

        stay_query_pair_feature = torch.cat([camera_pair_feature, region_feature], dim=-1)
        stay_query_pair_feature = stay_query_pair_feature * pool_mask.repeat(1, 1, stay_query_pair_feature.shape[-1])
        return stay_query_pair_feature

    def forward(self, camera_traj_data, camera_assign_mat, context_data, candidate_region, \
                grid_feature_map, camera_pair_feature, candidate_region_feature, stay_label):
        camera_pair_fea, binary_classification_label, camera_pair_speed = self.prepare_camera_pair(camera_traj_data, context_data, camera_assign_mat, camera_pair_feature, stay_label)
        region_feature, pair_candidate_region, where_stay_label = self.recall_candidate_grid(candidate_region, grid_feature_map,
                                                                      candidate_region_feature,
                                                                      stay_label)
        region_feature, pool_mask, src_key_padding_mask = self.prepare_canidate_grid(region_feature, pair_candidate_region)
                
        stay_query_pair_feature = self.prepare_query(camera_pair_fea, region_feature, pool_mask)
        stay_query_pair_feature = stay_query_pair_feature.to(torch.float)
        stay_query_pair_feature = self.linear(stay_query_pair_feature)
        stay_query_pair_rep = self.set_wise_encoder(stay_query_pair_feature, None, src_key_padding_mask)
        return camera_pair_speed, binary_classification_label, stay_query_pair_rep, where_stay_label, pool_mask
