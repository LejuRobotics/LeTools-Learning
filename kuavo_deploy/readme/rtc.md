# 如何使用inference time rtc 与 training time rtc
目前适配的模型包括 OpenPI、Kauvo-VLA、LingBot-VLA v2 和 GR00T N1.7

## inference time rtc
inference time rtc 无需训练，可直接在推理时使用

需修改deploy.yaml里的以下相关配置

```
  async_inference: true  # rtc需保持异步推理开启
  actions_per_chunk: null  
  chunk_size_threshold: 0.5  # 剩余动作比例 <= 此值时预取下一块
  aggregate_fn_name: exp  
  rtc_enabled: true  
  rtc_mode: guided  # guided：现有权重通过 VJP 引导，inference rtc；trained：需重新进行 RTC 训练, training rtc
  rtc_execution_horizon: 10  # 前缀引导长度，决定了inference rtc的前缀引导长度，根据实际推理消耗步数设置，需要小于当前chunk剩余的动作步数
  rtc_prefix_attention_schedule: exp  
  rtc_max_guidance_weight: 10.0
  rtc_initial_delay_steps: 1
```

## training time rtc
training time rtc 需要重新训练模型

使用时仅需将rtc_training_max_delay设置为非0值，具体数值建议比实际推理消耗步数大1-2步

对应配置文件在这几个模型的地址是

- `kuavo_model/external_models/lingbot-vla-v2/configs/vla/kuavo/kuavo_real.yaml`
- `kuavo_model/external_models/openpi/src/openpi/training/config.py`
- `kuavo_model/external_models/gr00tn1d7/gr00t/configs/finetune_config.py`

参数配置

```
  async_inference: true  # rtc需保持异步推理开启
  actions_per_chunk: null 
  chunk_size_threshold: 0.5  # 剩余动作比例 <= 此值时预取下一块
  aggregate_fn_name: exp  
  rtc_enabled: true  
  rtc_mode: trained  # guided：现有权重通过 VJP 引导，inference rtc；trained：需重新进行 RTC 训练, training rtc
  rtc_execution_horizon: 10  # 前缀引导长度，决定了inference rtc的前缀引导长度，根据实际推理消耗步数设置，需要小于当前chunk剩余的动作步数
  rtc_prefix_attention_schedule: exp  
  rtc_max_guidance_weight: 10.0 
  rtc_initial_delay_steps: 1
```