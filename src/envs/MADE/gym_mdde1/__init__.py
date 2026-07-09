import gym
import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='LoRaEnv-v1',                   # id for the old environment
    entry_point='gym_mdde1.envs.multi_lora:MultiFlyingLoRaEnv',    
)

# Association-scheme ablation env
# accepts env_args.assoc_mode=<gain|distance|random|fixed>.
register(
    id='LoRaEnvAssociationSchemeAblationStudy-v3',
    entry_point='gym_mdde1.envs.association_ablation_study:MultiFlyingLoRaEnv',
)

# Optimization-parameter ablation env 
# env_args.opt_mode=<full|fix_sf|fix_tp|fix_position|fix_all>.
register(
    id='LoRaEnvOptimizationParamAblationStudy-v4',
    entry_point='gym_mdde1.envs.optimization_param_ablation_study:MultiFlyingLoRaEnv',
)




