import torch

from components import *

class SAInfplus(BaseModel):
    def __init__(self, grid_feat_num,
                 num_camera, camera_embed_size,
                 num_user, user_embed_size,
                 num_grid, grid_embed_size,
                 minute_embed_size, weekofday_embed_size, weather_embed_size,
                 vehicle_type_embed_size, vehicle_color_embed_size, plate_color_embed_size,
                 hidden_dim, edge_index, drop_edge_rate,
                 drop_sequence_rate, drop_grid_rate, max_len):
        super(SAInfplus, self).__init__()
        self.num_grid = num_grid
        self.hidden_dim = hidden_dim
        self.edge_index = torch.tensor(edge_index).cuda()
        self.drop_edge_rate = drop_edge_rate

        # grid (gid) embedding
        self.grid_embedding = nn.Embedding(num_grid+1, grid_embed_size) # 0是padding
        self.grid_embedding.requires_grad_(True)

        # camera (cid) embedding
        self.camera_embedding = nn.Embedding(num_camera+1, camera_embed_size) # 0是padding
        self.camera_embedding.requires_grad_(True)

        # user embedding
        self.user_embedding = nn.Embedding(num_user+1, user_embed_size)
        self.user_embedding.requires_grad_(True)

        # weather
        self.weather_embedding = nn.Embedding(10, weather_embed_size)
        self.weather_embedding.requires_grad_(True)

        # minute embedding
        self.minute_embedding = nn.Embedding(1440+1, minute_embed_size) # 0是padding
        self.minute_embedding.requires_grad_(True)

        # weekofday embedding
        self.weekofday_embedding = nn.Embedding(7, weekofday_embed_size)
        self.weekofday_embedding.requires_grad_(True)

        # vehicle type embedding
        # self.vehicle_type_embedding = nn.Embedding(43+1, vehicle_type_embed_size)  # 0是padding
        # self.vehicle_type_embedding.requires_grad_(True)

        # vehicle color embedding
        # self.vehicle_color_embedding = nn.Embedding(14+1, vehicle_color_embed_size) # 0是padding
        # self.vehicle_color_embedding.requires_grad_(True)

        # plate color embedding
        # self.plate_color_embedding = nn.Embedding(8+1, plate_color_embed_size) # 0 是padding
        # self.plate_color_embedding.requires_grad_(True)

        # camera record transformation
        # cllx，csys，cpys，minute, cid, gid 不含timestamp,不用标准化
        # camera_record_feat_num = vehicle_type_embed_size + vehicle_color_embed_size + plate_color_embed_size \
        #                          + minute_embed_size + camera_embed_size + grid_embed_size
        camera_record_feat_num = minute_embed_size + camera_embed_size + grid_embed_size
        self.camera_record_linear = nn.Linear(camera_record_feat_num, hidden_dim)

        # context transformation
        # usr_id, week, weather, temperature
        context_feat_num = user_embed_size + weekofday_embed_size + weather_embed_size + 1
        self.context_linear = nn.Linear(context_feat_num, hidden_dim)

        # camera_sequence_encoder
        # self.graph_encoder = GraphEncoder(grid_embed_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_len+1, hidden_dim)
        self.token_linear = nn.Linear(hidden_dim, hidden_dim)
        self.camera_sequence_encoder = TransformerModel(hidden_dim, 8, hidden_dim, 2, drop_sequence_rate)
        self.pair_linear = nn.Linear(3+2*hidden_dim, hidden_dim)

        # candidate_region_encoder
        self.grid_linear = nn.Linear(grid_feat_num+4, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.candidate_region_encoder = TransformerModel(hidden_dim, 8, hidden_dim, 2, drop_grid_rate)
        self.mobility_spatial_interactor = Attention(hidden_dim, hidden_dim, hidden_dim)
        # self.mobility_spatial_interactor = MultiHeadAttention(hidden_dim, hidden_dim, hidden_dim, 4)

        self.whether_stay_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(True), nn.Linear(64, 1)) #
        self.where_stay_head = nn.Sequential(nn.Linear(hidden_dim*2, 1)) # , nn.ReLU(True), nn.Linear(64, 1)


    def encode_context(self, context_data):
        # usr_id, week, weather, temperature
        # size * max_len * 4

        user_emb = self.user_embedding(context_data[:, 0].long()) # size * max_len * 1
        weekofday_emb = self.weekofday_embedding(context_data[:, 1].long())
        weather_emb = self.weather_embedding(context_data[:, 2].long())
        temperature_value = context_data[:, 3].unsqueeze(-1)

        context_feat = torch.cat([user_emb, weekofday_emb, weather_emb, temperature_value], dim=-1)

        context_rep = self.context_linear(context_feat)

        return context_rep

    def encode_camera_sequence(self, camera_traj_data, camera_assign_mat, context_data):
        # cllx，cpys，csys，minute，timestamp，cid, gid

        # vehicle type embedding
        timestamp_feature = camera_traj_data[:, :, 4]
        camera_traj_data = camera_traj_data + 1 # padding位置从-1变为0，统一embedding的位置一样，方便后面mask

        # vehicle_type_emb = self.vehicle_type_embedding(camera_traj_data[:, :, 0].long())
        # plate_color_emb = self.plate_color_embedding(camera_traj_data[:, :, 1].long())
        # vehicle_color_emb = self.vehicle_color_embedding(camera_traj_data[:, :, 2].long())
        minute_emb = self.minute_embedding(camera_traj_data[:, :, 3].long())
        camera_emb = self.camera_embedding(camera_traj_data[:, :, 5].long())
        grid_emb = self.grid_embedding(camera_traj_data[:, :, 6].long())

        # todo GAT更新grid_embed
        # todo 分配矩阵 N*M 更新 camera_embed

        # camera_traj_feature = torch.cat([vehicle_type_emb,vehicle_color_emb,plate_color_emb,\
        #                                  minute_emb, camera_emb, grid_emb], dim=-1) # batch_size * max_len * (64*4)

        camera_traj_feature = torch.cat([minute_emb, camera_emb, grid_emb], dim=-1) # batch_size * max_len * (64*4)

        camera_traj_rep = self.camera_record_linear(camera_traj_feature)

        # context 作为cls token 在序列维度拼接到camera_traj_rep上
        context_rep = self.encode_context(context_data)
        camera_sequence_rep = torch.cat([context_rep.unsqueeze(1), camera_traj_rep], dim=1)
        # 对应的camera_traj_mat也要重新更新
        camera_assign_mat = torch.cat([camera_assign_mat[:, 0].unsqueeze(-1), camera_assign_mat], dim=1)

        position = torch.arange(camera_sequence_rep.shape[1]).long().cuda()
        pos_emb = position.unsqueeze(0).repeat(camera_sequence_rep.shape[0], 1)  # (S,) -> (B, S)
        pos_emb = self.position_embedding(pos_emb)

        camera_sequence_rep = camera_sequence_rep + pos_emb

        # todo 计算 interval_mat  用于attention计算
        # todo 集成 pair 之间的特征 主要是时间

        src_key_padding_mask = (camera_assign_mat == self.num_grid)
        pool_mask = (1 - src_key_padding_mask.int()).unsqueeze(-1)  # 0 为padding位

        camera_sequence_rep = self.token_linear(camera_sequence_rep)
        camera_sequence_rep = self.camera_sequence_encoder(camera_sequence_rep, None, src_key_padding_mask)  # mask 被在这里处理，mask不参与计算attention
        camera_sequence_rep = torch.where(\
            torch.isnan(camera_sequence_rep), torch.full_like(camera_sequence_rep, 0), camera_sequence_rep)  # 将nan变为0,防止溢出
        camera_sequence_rep = camera_sequence_rep * pool_mask.repeat(1, 1, camera_sequence_rep.shape[-1])  # (batch_size,max_len,feat_num)

        return camera_sequence_rep # batch * max_len * hidden_size


    def traj2pair(self, camera_sequence_rep, camera_assign_mat, camera_pair_feature, stay_label):
        # 将以traj为单位的组织方式转化成以pair为单位的组织方式，traj中的padding处去除
        camera_pair_rep_list = []
        binary_classification_label = []
        for i in range(camera_assign_mat.shape[0]):
            for j in range(camera_assign_mat.shape[1]-1):
                if camera_assign_mat[i][j] == self.num_grid or camera_assign_mat[i][j+1] == self.num_grid:
                    continue
                else:
                    #camera_assign_mat中第 j 和 j+1 的记录对 对应的是camera_pair_feature 中的第j个
                    pair_rep = torch.cat(\
                        [camera_pair_feature[i][j], camera_sequence_rep[i][j], camera_sequence_rep[i][j+1]],dim=-1)
                    pair_rep = self.pair_linear(pair_rep)
                    camera_pair_rep_list.append(pair_rep.unsqueeze(0))

                num_stay_grid = (stay_label[i][j] != self.num_grid).long().sum().item()
                if num_stay_grid == 0:
                    binary_classification_label.append(0)
                else:
                    binary_classification_label.append(1)

        camera_pair_rep = torch.cat(camera_pair_rep_list, dim=0)
        binary_classification_label = torch.tensor(binary_classification_label)
        return camera_pair_rep, binary_classification_label


    # version 1: 使用驻留点的空间分布生成单个候选集
    def recall_candidate_grid(self, candidate_region, grid_feature_map, candidate_region_feature, stay_label):
        # 将以traj为单位的组织方式转化成以pair为单位的组织方式，traj中的padding处去除
        # 对于候选集已经预先选定好的，应该放到 dataloader 中处理 这样计算效率高一些
        # num_traj * (max_len-1) * max_stay_grid
        # 64*19*256

        num_feature = len(list(grid_feature_map.values())[0])

        region_feature_list = []
        where_stay_label_list = []
        for i in range(candidate_region.shape[0]): # -1为padding位
            for j in range(candidate_region.shape[1]):
                candidate_region_set = candidate_region[i][j]
                candidate_region_set_feature = candidate_region_feature[i][j]
                if (candidate_region_set != self.num_grid).long().sum().item() == 0: # 在构造candidate_region时，padding的位置的候选集全部都为num_grid
                    continue # padding pair位置pass
                else: # 非padding pair
                    this_region_feature_list = []
                    for grid in candidate_region_set:
                        try:
                            this_region_feature_list.append(grid_feature_map[int(grid)]) # candidate_region 在cpu上，int只会取值不会发生设备转移
                        except:
                            this_region_feature_list.append([float('nan')]*num_feature)

                    this_region_feature = torch.tensor(this_region_feature_list, dtype=torch.float)# max_stay_grid*23
                    this_region_feature = torch.cat([this_region_feature,candidate_region_set_feature],dim=-1)

                    # this_region_feature = torch.cat(this_region_feature_list, dim=0)

                    region_feature_list.append(this_region_feature.unsqueeze(0))

                    candidate_region_set = candidate_region_set[candidate_region_set != self.num_grid]

                    stay_region = stay_label[i][j]
                    stay_region = stay_region[stay_region != self.num_grid]

                    label_idx = torch.where((candidate_region_set[..., None] == stay_region).any(-1))[0]
                    label = torch.zeros(candidate_region.shape[-1])
                    label[label_idx] = 1

                    where_stay_label_list.append(label.unsqueeze(0))

        region_feature = torch.cat(region_feature_list, dim=0).cuda() # num_pair * max_stay_grid * 23
        where_stay_label = torch.cat(where_stay_label_list, dim=0).cuda()

        return region_feature, where_stay_label

    # version 2: 使用stay query pair的信息生成多个候选集
    # def multi_path_recall_canidate_grid(self, grid_mat, grid_feature, camera_pair_rep, num_heads=5):
    #
    #     assert self.hidden_dim % num_heads == 0
    #     # 调用多注意力机制生成num_heads个1*max_candidate_grids的向量。每个向量，是一个候选集。
    #     return grid_masked_mat  # batch * num_heads * max_candidate_grids

    def encode_canidate_grid(self, region_feature):

        src_key_padding_mask = torch.isnan(region_feature)[:, :, 0]
        pool_mask = (1 - src_key_padding_mask.int()).unsqueeze(-1)  # 0 为padding位

        region_feature = torch.nan_to_num(region_feature, nan=0.0) # norm之前，否则mean和std会计算为0，使linear参数归零
        region_rep = self.grid_linear(region_feature)
        region_rep = self.norm(region_rep)  # B * N * D

        region_rep = self.candidate_region_encoder(region_rep, None,
                                                 src_key_padding_mask)  # mask 被在这里处理，mask不参与计算attention
        region_rep = region_rep * pool_mask.repeat(1, 1, region_rep.shape[
            -1])
        return region_rep, pool_mask

    def multihead_st_query(self, camera_pair_rep, region_rep, pool_mask):

        camera_pair_rep = camera_pair_rep.unsqueeze(1).expand_as(region_rep)
        camera_pair_rep = self.mobility_spatial_interactor(camera_pair_rep, region_rep)

        stay_query_pair_rep = torch.cat([camera_pair_rep, region_rep], dim=-1)
        stay_query_pair_rep = stay_query_pair_rep * pool_mask.repeat(1, 1, stay_query_pair_rep.shape[-1])

        return stay_query_pair_rep

    def forward(self, camera_traj_data, camera_assign_mat, context_data, candidate_region, \
                grid_feature_map, camera_pair_feature, candidate_region_feature, stay_label):

        camera_sequence_rep = self.encode_camera_sequence(camera_traj_data, camera_assign_mat, context_data)
        camera_pair_rep, binary_classification_label = self.traj2pair(camera_sequence_rep, \
                                                                      camera_assign_mat, camera_pair_feature, stay_label)
        region_feature, where_stay_label = self.recall_candidate_grid(candidate_region, grid_feature_map, candidate_region_feature, stay_label) # 输入都是在cpu上的操作
        region_rep, pool_mask = self.encode_canidate_grid(region_feature)
        stay_query_pair_rep = self.multihead_st_query(camera_pair_rep, region_rep, pool_mask)
        return camera_pair_rep, binary_classification_label, stay_query_pair_rep, where_stay_label, pool_mask

