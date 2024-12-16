- 会议版的工作证明了什么？
    1. 证明了SAInf这种分层框架的优越性
    2. 证明了深度学习在ranking任务上的优越性

- RInf 对于每个pair不检测驻留事件，在pair两个摄像头的中心位置周围5km内随机选k个格子
- SHInf 对于每个pair不检测驻留事件，在pair两个摄像头的中心位置周围5km内按照历史出现的频率选k个格子
- VHInf 对于每个pair使用经验速度检测驻留事件，在pair两个摄像头的中心位置周围按照旅行时间和经验速度计算可达范围，在该范围内按照历史出现的频率选k个格子
- VSHInf 对于每个pair使用经验速度检测驻留事件，在pair两个摄像头的中心位置周围5km内按照历史出现的频率选k个格子

- SAInf 分层框架：stay event detection (statistical method) -> candidate region generation (statistical method) -> candidate region ranking (NN)

    - SHS
        - 与SHInf, VHInf, VSHInf 一致
        - 在候选集中按照历史频率选择top-k的结果 
    - STHS
        - 按照驻留的开始时间构成三维的历史频率lookup, 选择topk的结果
    - RNN
        - 输入point-wise的[query1,canididate_region1]，有单独的分类头
    - Transformer
        - 输入point-wise的[query1,canididate_region1]，有单独的分类头

- SAInfplus end2end：stay event detection 和 candidate region ranking 联合训练 + candidate region generation (statistical method)


