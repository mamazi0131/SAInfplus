from components import *

class framework:
    def __init__(self,num_grid,
                 speed_threshold,
                 grid_freq_weight):
        self.speed_threshold = speed_threshold
        # num_grid 是 padding，和model中的gird_embedding不同，
        # 后者是作用在camera_traj_data上的，而前者是作用在camera_assign_mat上的
        self.grid_freq = nn.Embedding(num_grid+1, 1) # num_grid 是 padding
        self.grid_freq.requires_grad_(False)
        self.grid_freq.weight = torch.nn.Parameter(grid_freq_weight.unsqueeze(-1))

    def stay_event_detection(self, camera_pair_speed):
        # camera_record_pair 是指每两个连续的监控摄像头的速度
        return (camera_pair_speed <= self.speed_threshold)

    def candidate_region_ranking(self, pair_candidate_region):
        # candidate_region 是指基于stay_query_pair生成的候选区域
        candidate_region_freq = self.grid_freq(pair_candidate_region)
        # candidate_region 按照 candidate_region_freq 排序
        # sorted_index = torch.argsort(candidate_region_freq,dim=2,descending=True)
        # candidate_region_rank_result = torch.gather(candidate_region,dim=2,index=sorted_index)
        return candidate_region_freq # num_pair * max_candidate_region
    
    def stay_area_inference(self, camera_pair_speed, candidate_region):
        whether_stay_pred = self.stay_event_detection(camera_pair_speed)
        candidate_region_freq = self.candidate_region_ranking(candidate_region)
        return whether_stay_pred, candidate_region_freq

# RInf 对于每个pair不检测驻留事件，在每pair两个摄像头的中心位置周围5km内随机选k个格子
## grid_freq padding为-1，其他为1
## speed_threshold 为 0m/s
## distance_threshold 为 5000

# SHInf 对于每个pair不检测驻留事件，在pair两个摄像头的中心位置周围5km内按照历史出现的频率选k个格子
## grid_freq padding为-1，其他为freq
## speed_threshold 为 0m/s
## distance_threshold 为 5000m

# VHInf 对于每个pair使用经验速度检测驻留事件，在pair两个摄像头的中心位置周围按照旅行时间和经验速度计算可达范围，在该范围内按照历史出现的频率选k个格子
## grid_freq padding为-1，其他为freq
## speed_threshold 为 3m/s
## distance_threshold 为 动态  

# VSHInf 对于每个pair使用经验速度检测驻留事件，在pair两个摄像头的中心位置周围按照旅行时间和经验速度计算可达范围，在该范围内按照历史出现的频率选k个格子
## grid_freq padding为-1，其他为freq
## speed_threshold 为 3m/s
## distance_threshold 为 动态  
