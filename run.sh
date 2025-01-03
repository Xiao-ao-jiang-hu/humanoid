# export TASK="h1hand-truck-v0"
export TASK="h1hand-walk-v0"
# export TASK="h1hand-bookshelf_simple-v0"
# export TASK="h1hand-powerlift-v0"

export MUJOCO_GL=egl
# CUDA_VISIBLE_DEVICES=3 python ./jaxrl_m/examples/mujoco/run_mujoco_sac.py --env_name ${TASK} --wandb_entity wst22 --seed 0

CUDA_VISIBLE_DEVICES=7 python -m embodied.agents.dreamerv3.train --configs humanoid_benchmark --run.wandb True --run.wandb_entity wst22 --method dreamer --logdir logs/${TASK} --task humanoid_${TASK} --seed 0

# CUDA_VISIBLE_DEVICES=1 python -m tdmpc2.train disable_wandb=False wandb_entity=wst22 exp_name=tdmpc task=humanoid_${TASK} seed=0

# CUDA_VISIBLE_DEVICES=4 python ./ppo/run_sb3_ppo.py --env_name ${TASK} --wandb_entity wst22 --seed 0