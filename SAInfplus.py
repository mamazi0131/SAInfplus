from components import *
import numpy as np
from scipy.interpolate import interp1d

class SAInfplus(BaseModel):
    def __init__(self, grid_feat_num,
                 num_camera, camera_embed_size,
                 num_user, user_embed_size,
                 num_grid, grid_embed_size,
                 minute_embed_size, weekofday_embed_size, weather_embed_size,
                 vehicle_type_embed_size, vehicle_color_embed_size, plate_color_embed_size,
                 hidden_dim, drop_grid_rate, max_len, edge_index):
        super(SAInfplus, self).__init__()
        self.num_grid = num_grid
        self.hidden_dim = hidden_dim

        # 1. pair encoding
        # 1.1 grid embedding （share with 2）
        self.grid_embedding = nn.Embedding(num_grid+1, grid_embed_size) # 0 is padding
        self.grid_embedding.requires_grad_(True)

        # 1.2 camera embedding
        self.camera_embedding = nn.Embedding(num_camera+1, camera_embed_size) # 0 is padding
        self.camera_embedding.requires_grad_(True)

        # 1.3 user embedding
        self.user_embedding = nn.Embedding(num_user+1, user_embed_size)
        self.user_embedding.requires_grad_(True)

        # 1.4 vehicle type embedding
        self.vehicle_type_embedding = nn.Embedding(43+1, vehicle_type_embed_size)  # 0 is padding
        self.vehicle_type_embedding.requires_grad_(True)

        # 1.5 vehicle color embedding
        self.vehicle_color_embedding = nn.Embedding(14+1, vehicle_color_embed_size) # 0 is padding
        self.vehicle_color_embedding.requires_grad_(True)

        # 1.6 plate color embedding
        self.plate_color_embedding = nn.Embedding(8+1, plate_color_embed_size) # 0 is padding
        self.plate_color_embedding.requires_grad_(True)

        # 1.7 weather
        self.weather_embedding = nn.Embedding(10, weather_embed_size)
        self.weather_embedding.requires_grad_(True)

        # 1.8 minute embedding
        self.minute_embedding = nn.Embedding(1440+1, minute_embed_size) # 0 is padding
        self.minute_embedding.requires_grad_(True)

        # 1.9 weekofday embedding
        self.weekofday_embedding = nn.Embedding(7, weekofday_embed_size)
        self.weekofday_embedding.requires_grad_(True)

        # 1.10 camera encoding
        self.graph_encoder = GraphEncoder(camera_embed_size, hidden_dim, drop_grid_rate)
        self.edge_index = edge_index   

        # 1.11 record encoding
        camera_record_feat_num = vehicle_type_embed_size + vehicle_color_embed_size + plate_color_embed_size \
                                + minute_embed_size + camera_embed_size + grid_embed_size # 64*6

        self.camera_record_linear = nn.Linear(camera_record_feat_num, hidden_dim)

        # 1.12 context encoding
        context_feat_num = user_embed_size + weekofday_embed_size + weather_embed_size + 1
        self.context_linear = nn.Linear(context_feat_num, hidden_dim)

        # 1.13 movement encoding
        self.pair_camera_linear = nn.Linear(3, hidden_dim)

        # 2. candidate_region_encoder
        self.grid_linear = nn.Linear(4, grid_embed_size)
        self.norm = nn.LayerNorm(hidden_dim)
        self.candidate_region_encoder = TransformerModel(hidden_dim, 8, hidden_dim, 2, drop_grid_rate)

        # 3. dual task inference network
        self.mmoe = MMOE(4*hidden_dim,hidden_dim,n_expert=4,n_task=2,use_gate=True)
        # 3.1 selection head
        self.mobility_spatial_interactor = PointerNet(hidden_dim, hidden_dim, hidden_dim)       
        # 3.2 detection head
        self.whether_stay_head = nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(True), nn.Linear(64, 1)) #

             
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

    def encode_camera_pair(self, camera_traj_data, context_data, camera_assign_mat, camera_pair_feature, stay_label):
        # cllx，cpys，csys，minute，timestamp，cid, gid

        camera_traj_data = camera_traj_data + 1 # padding from -1 to 0, easy to mask

        vehicle_type_emb = self.vehicle_type_embedding(camera_traj_data[:, :, 0].long())
        plate_color_emb = self.plate_color_embedding(camera_traj_data[:, :, 1].long())
        vehicle_color_emb = self.vehicle_color_embedding(camera_traj_data[:, :, 2].long())
        minute_emb = self.minute_embedding(camera_traj_data[:, :, 3].long())
        grid_emb = self.grid_embedding(camera_traj_data[:, :, 6].long())

        

        # update cid
        padding_vec = self.camera_embedding.weight[0].unsqueeze(0)
        camera_node_rep = self.graph_encoder(self.camera_embedding.weight[1:,:], self.edge_index)
        lookup_table = torch.concat([padding_vec,camera_node_rep],dim=0)
        batch_size, max_len = camera_traj_data.shape[0], camera_traj_data.shape[1]
        camera_emb = torch.index_select(lookup_table,0,camera_traj_data[:, :, 5].long().view(-1)).view(batch_size,max_len,-1)

        camera_traj_feature = torch.cat([vehicle_type_emb,vehicle_color_emb,plate_color_emb,\
                                         minute_emb, camera_emb, grid_emb], dim=-1) # batch_size * max_len * (64*4)
        
        
        camera_traj_rep = self.camera_record_linear(camera_traj_feature)
        context_rep = self.encode_context(context_data)

        camera_pair_rep_list = []
        binary_classification_label_list = []
        camera_pair_speed_list = []
        for i in range(camera_assign_mat.shape[0]):
            for j in range(camera_assign_mat.shape[1] - 1):
                if camera_assign_mat[i][j] == self.num_grid or camera_assign_mat[i][j + 1] == self.num_grid:
                    continue
                else:
                    pair_camera_rep = self.pair_camera_linear(camera_pair_feature[i][j])
                    pair_rep = torch.cat( \
                        [context_rep[i], pair_camera_rep, camera_traj_rep[i][j], camera_traj_rep[i][j + 1]], dim=-1)
                    
                    camera_pair_rep_list.append(pair_rep.unsqueeze(0))

                    camera_pair_speed = camera_pair_feature[i][j][-1]
                    camera_pair_speed_list.append(camera_pair_speed)

                num_stay_grid = (stay_label[i][j] != self.num_grid).long().sum().item()
                if num_stay_grid == 0:
                    binary_classification_label_list.append(0)
                else:
                    binary_classification_label_list.append(1)

                    
        camera_pair_rep = torch.cat(camera_pair_rep_list, dim=0)
        towers = self.mmoe(camera_pair_rep)

        whether_camera_pair_rep = towers[0]
        where_camera_pair_rep = towers[1]
                
        binary_classification_label = torch.tensor(binary_classification_label_list)
        return whether_camera_pair_rep, where_camera_pair_rep, binary_classification_label, camera_pair_speed

    def recall_candidate_grid(self, candidate_region, grid_feature_map, candidate_region_feature, stay_label):
        # num_traj * (max_len-1) * max_stay_grid
        # 64*19*256

        num_feature = len(list(grid_feature_map.values())[0])

        pair_candidate_region_list = []

        region_feature_list = []
        where_stay_label_list = []
        for i in range(candidate_region.shape[0]):  # -1 is padding
            for j in range(candidate_region.shape[1]):
                candidate_region_set = candidate_region[i][j]
                candidate_region_set_feature = candidate_region_feature[i][j]
                if (
                        candidate_region_set != self.num_grid).long().sum().item() == 0: 
                    continue  # padding pair is pass
                else:
                    pair_candidate_region_list.append(candidate_region_set.unsqueeze(0))
                    this_region_feature_list = []
                    for grid in candidate_region_set:
                        try:
                            this_region_feature_list.append(
                                grid_feature_map[int(grid)]) 
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

    def encode_canidate_grid(self, region_feature, pair_candidate_region):

        src_key_padding_mask = torch.isnan(region_feature)[:, :, 0]
        pool_mask = (1 - src_key_padding_mask.int()).unsqueeze(-1)  # 0 is padding

        region_feature = torch.nan_to_num(region_feature, nan=0.0) 
        region_rep = self.grid_linear(region_feature)

        region_emb = self.grid_embedding(pair_candidate_region)
        region_rep = torch.cat([region_rep, region_emb], dim=-1)

        region_rep = self.norm(region_rep)  # B * N * D
        region_rep = self.candidate_region_encoder(region_rep, None,
                                                 src_key_padding_mask) 
        region_rep = region_rep * pool_mask.repeat(1, 1, region_rep.shape[
            -1])
        return region_rep, pool_mask

    def multihead_st_query(self, camera_pair_rep, region_rep, pool_mask):

        camera_pair_rep = camera_pair_rep.unsqueeze(1).expand_as(region_rep)
        p = self.mobility_spatial_interactor(camera_pair_rep, region_rep)
        p = p.sum(dim=-2)
        p = p.unsqueeze(-1) * pool_mask
        return p

    def forward(self, camera_traj_data, camera_assign_mat, context_data, candidate_region, \
                grid_feature_map, camera_pair_feature, candidate_region_feature, stay_label):
        whether_camera_pair_rep, where_camera_pair_rep, binary_classification_label,_= \
            self.encode_camera_pair(camera_traj_data, context_data, camera_assign_mat, camera_pair_feature, stay_label)

        region_feature, pair_candidate_region, where_stay_label = self.recall_candidate_grid(candidate_region, grid_feature_map,
                                                                      candidate_region_feature,
                                                                      stay_label) 
        region_rep, pool_mask = self.encode_canidate_grid(region_feature, pair_candidate_region)
        where_stay_pred = self.multihead_st_query(where_camera_pair_rep, region_rep, pool_mask)
        return whether_camera_pair_rep, binary_classification_label, where_stay_pred, where_stay_label, pool_mask

    