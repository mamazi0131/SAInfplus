from components import *
import numpy as np
from scipy.interpolate import interp1d

class SAInf(BaseModel):
    def __init__(self, grid_feat_num,
                 num_camera, camera_embed_size,
                 num_user, user_embed_size,
                 num_grid, grid_embed_size,
                 minute_embed_size, weekofday_embed_size, weather_embed_size,
                 vehicle_type_embed_size, vehicle_color_embed_size, plate_color_embed_size,
                 hidden_dim, drop_grid_rate, threshold):
        super(SAInf, self).__init__()
        self.num_grid = num_grid
        self.hidden_dim = hidden_dim
        self.threshold = threshold

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
        self.vehicle_type_embedding = nn.Embedding(43+1, vehicle_type_embed_size)  # 0是padding
        self.vehicle_type_embedding.requires_grad_(True)

        # vehicle color embedding
        self.vehicle_color_embedding = nn.Embedding(14+1, vehicle_color_embed_size) # 0是padding
        self.vehicle_color_embedding.requires_grad_(True)

        # plate color embedding
        self.plate_color_embedding = nn.Embedding(8+1, plate_color_embed_size) # 0 是padding
        self.plate_color_embedding.requires_grad_(True)

        # camera record transformation
        # cllx，csys，cpys，minute, cid, gid 不含timestamp,不用标准化
        camera_record_feat_num = vehicle_type_embed_size + vehicle_color_embed_size + plate_color_embed_size \
                                 + minute_embed_size + camera_embed_size + grid_embed_size # 64*6

        # camera_record_feat_num = minute_embed_size + camera_embed_size + grid_embed_size  # 64*6

        self.camera_record_linear = nn.Linear(camera_record_feat_num, hidden_dim)

        # context transformation
        # usr_id, week, weather, temperature
        context_feat_num = user_embed_size + weekofday_embed_size + weather_embed_size + 1
        self.context_linear = nn.Linear(context_feat_num, hidden_dim)

        # camera_piar_encoder
        # self.pair_linear = nn.Linear(3+2*hidden_dim, hidden_dim)
        self.pair_linear = nn.Linear(3+3*hidden_dim, hidden_dim)

        # candidate_region_encoder
        self.grid_linear = nn.Linear(grid_feat_num+4, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.candidate_region_encoder = TransformerModel(hidden_dim, 8, hidden_dim, 2, drop_grid_rate)

        # spatial_temoporal_query
        self.mobility_spatial_interactor = Attention(hidden_dim, hidden_dim, hidden_dim)

        self.where_stay_head = nn.Sequential(nn.Linear(hidden_dim*2, 64), nn.ReLU(True), nn.Linear(64, 1))

    def stay_evenet_detection(self, camera_pair_speed):
        # camera_pair_speed 维度为1 指两个pair间的speed
        return (camera_pair_speed <= self.threshold).long()

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

        camera_traj_data = camera_traj_data + 1 # padding位置从-1变为0，统一embedding的位置一样，方便后面mask

        vehicle_type_emb = self.vehicle_type_embedding(camera_traj_data[:, :, 0].long())
        plate_color_emb = self.plate_color_embedding(camera_traj_data[:, :, 1].long())
        vehicle_color_emb = self.vehicle_color_embedding(camera_traj_data[:, :, 2].long())
        minute_emb = self.minute_embedding(camera_traj_data[:, :, 3].long())
        camera_emb = self.camera_embedding(camera_traj_data[:, :, 5].long())
        grid_emb = self.grid_embedding(camera_traj_data[:, :, 6].long())

        camera_traj_feature = torch.cat([vehicle_type_emb,vehicle_color_emb,plate_color_emb,\
                                         minute_emb, camera_emb, grid_emb], dim=-1) # batch_size * max_len * (64*4)
        # camera_traj_feature = torch.cat([minute_emb, camera_emb, grid_emb], dim=-1) # batch_size * max_len * (64*4)

        camera_traj_rep = self.camera_record_linear(camera_traj_feature)
        context_rep = self.encode_context(context_data)

        # todo 先单独过linear layer再拼接
        # todo 先拼接，再一起过linear layer

        camera_pair_rep_list = []
        binary_classification_label_list = []
        camera_pair_speed_list = []
        for i in range(camera_assign_mat.shape[0]):
            for j in range(camera_assign_mat.shape[1] - 1):
                if camera_assign_mat[i][j] == self.num_grid or camera_assign_mat[i][j + 1] == self.num_grid:
                    continue
                else:
                    # camera_assign_mat中第 j 和 j+1 的记录对 对应的是camera_pair_feature 中的第j个
                    pair_rep = torch.cat( \
                        [context_rep[i], camera_pair_feature[i][j], camera_traj_rep[i][j], camera_traj_rep[i][j + 1]], dim=-1)
                    pair_rep = self.pair_linear(pair_rep)
                    camera_pair_rep_list.append(pair_rep.unsqueeze(0))

                    camera_pair_speed = camera_pair_feature[i][j][-1]
                    camera_pair_speed_list.append(camera_pair_speed)

                num_stay_grid = (stay_label[i][j] != self.num_grid).long().sum().item()
                if num_stay_grid == 0:
                    binary_classification_label_list.append(0)
                else:
                    binary_classification_label_list.append(1)

        camera_pair_rep = torch.cat(camera_pair_rep_list, dim=0)
        binary_classification_label = torch.tensor(binary_classification_label_list)
        camera_pair_speed = torch.tensor(camera_pair_speed_list)
        return camera_pair_rep, binary_classification_label, camera_pair_speed

    def recall_candidate_grid(self, candidate_region, grid_feature_map, candidate_region_feature, stay_label):
        # 将以traj为单位的组织方式转化成以pair为单位的组织方式，traj中的padding处去除
        # 对于候选集已经预先选定好的，应该放到 dataloader 中处理 这样计算效率高一些
        # num_traj * (max_len-1) * max_stay_grid
        # 64*19*256

        num_feature = len(list(grid_feature_map.values())[0])

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
                    this_region_feature_list = []
                    for grid in candidate_region_set:
                        try:
                            this_region_feature_list.append(
                                grid_feature_map[int(grid)])  # candidate_region 在cpu上，int只会取值不会发生设备转移
                        except:
                            this_region_feature_list.append([float('nan')] * num_feature)

                    this_region_feature = torch.tensor(this_region_feature_list, dtype=torch.float)  # max_stay_grid*23
                    this_region_feature = torch.cat([this_region_feature, candidate_region_set_feature], dim=-1)

                    # this_region_feature = torch.cat(this_region_feature_list, dim=0)

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

        return region_feature, where_stay_label

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
        camera_pair_rep, binary_classification_label,camera_pair_speed = self.encode_camera_pair(camera_traj_data, context_data, camera_assign_mat, camera_pair_feature, stay_label)
        region_feature, where_stay_label = self.recall_candidate_grid(candidate_region, grid_feature_map,
                                                                      candidate_region_feature,
                                                                      stay_label)  # 输入都是在cpu上的操作
        region_rep, pool_mask = self.encode_canidate_grid(region_feature)
        stay_query_pair_rep = self.multihead_st_query(camera_pair_rep, region_rep, pool_mask)
        return camera_pair_speed, binary_classification_label, stay_query_pair_rep, where_stay_label, pool_mask

class Detection_Threshold_Estimation():
    def __init__(self, stay_distribution, unstay_distribution, bins=10000):
        self.stay_distribution = np.sort(stay_distribution)
        self.unstay_distribution = np.sort(unstay_distribution)
        self.stay_cdf = 1.0 * np.arange(len(stay_distribution)) / float(len(stay_distribution) - 1)
        self.unstay_cdf = 1.0 * np.arange(len(unstay_distribution)) / float(len(unstay_distribution) - 1)
        self.bins = bins

    def estimate_threshold(self):
        min_bound = min(min(self.stay_distribution), min(self.unstay_distribution))
        max_bound = max(max(self.stay_distribution), max(self.unstay_distribution))
        bins = 10000
        bin_width = (max_bound - min_bound) / bins
        # 对齐数据点，使用线性插值
        interp_func1 = interp1d(self.stay_distribution, self.stay_cdf, kind='linear', fill_value='extrapolate')
        interp_func2 = interp1d(self.unstay_distribution, self.unstay_cdf, kind='linear', fill_value='extrapolate')

        # 定义新的CDF范围，例如0到1，可以根据实际情况调整
        new_range = np.linspace(min_bound, max_bound, bins)

        # 对齐后的数据点
        aligned_stay_cdf = interp_func1(new_range)
        aligned_unstay_cdf = interp_func2(new_range)

        # 处理nan值
        aligned_stay_cdf[np.isnan(aligned_stay_cdf)] = 0
        aligned_unstay_cdf[np.isnan(aligned_unstay_cdf)] = 0

        max_distance_index = np.argmax(np.abs(aligned_stay_cdf - aligned_unstay_cdf))
        threshold = min_bound + max_distance_index * bin_width
        return threshold