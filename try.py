

import ray
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import humanoid_bench
from tqdm import trange
# 启动 Ray
import multiprocessing
# 简单的神经网络模型

# 假设你使用一个简单的神经网络


class SimpleNN(torch.nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = torch.nn.Linear(151, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 512)
        self.fc4 = torch.nn.Linear(512, 61)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        x = self.act(self.fc4(x))
        return x


# class SimpleNN(torch.nn.Module):
#     def __init__(self):
#         super(SimpleNN, self).__init__()
#         self.fc1 = torch.nn.Linear(151, 1024)
#         self.fc2 = torch.nn.Linear(1024, 1024)
#         self.fc3 = torch.nn.Linear(1024, 1024)
#         self.fc4 = torch.nn.Linear(1024, 1024)
#         self.fc5 = torch.nn.Linear(1024, 1024)
#         self.fc6 = torch.nn.Linear(1024, 1024)
#         self.fc7 = torch.nn.Linear(1024, 1024)
#         self.fc8 = torch.nn.Linear(1024, 61)
#         self.act = nn.Tanh()

#     def forward(self, x):
#         x = self.act(self.fc1(x))
#         x = self.act(self.fc2(x))
#         x = self.act(self.fc3(x))
#         x = self.act(self.fc4(x))
#         x = self.act(self.fc5(x))
#         x = self.act(self.fc6(x))
#         x = self.act(self.fc7(x))
#         x = self.act(self.fc8(x))
#         return x


def worker(model, env_id, steps=1024):
    # 在工作进程中通过共享字典访问已创建的环境
    with torch.no_grad():
        env = gym.make(env_id)
        if env is None:
            print(f"Error: Environment {env_id} not found.")
            return

        state = env.reset()[0]
        state = torch.tensor(state, dtype=torch.float32)
        done = False
        total_reward = 0
        count_steps = 0
        count = 0
        for _ in range(steps):
            # print(state.shape)

            action = model(state)
            state, reward, done, _, _ = env.step(action)
            state = torch.tensor(state, dtype=torch.float32)
            total_reward += reward
            if done == True:
                count += 1
                state = env.reset()[0]
                state = torch.tensor(state, dtype=torch.float32)
                done = False

        return total_reward/count


def main():
    env_id = "h1hand-walk-v0"  # 选择的环境ID
    num_workers = 62         # 启动的worker进程数

    # 创建一个神经网络模型
    model = SimpleNN()

    # with multiprocessing.Manager() as manager:
    #     # 创建共享字典，存储环境的引用
    #     env_state_dict = manager.dict()

    #     # 主进程创建环境并将其保存到共享字典中
    #     create_env(env_id, env_state_dict)

    # 使用多进程池来启动 worker 进程
    with multiprocessing.Pool(num_workers) as pool:
        # 启动多个工作进程，每个进程使用共享的环境状态
        results = pool.starmap(
            worker, [(model, env_id, 1024)] * num_workers)
        print("Total rewards from each worker:", results)


if __name__ == "__main__":
    main()

# def sample_env(num_steps, id):
#     env = gym.make("h1hand-walk-v0")
#     print(env.action_space.shape, env.observation_space.shape)
#     state = env.reset()
#     done = False
#     trajectory = []
#     if id == 0:
#         loader = trange(num_steps)
#     else:
#         loader = range(num_steps)
#     for _ in loader:
#         if done:
#             state = env.reset()
#             done = False
#         action = env.action_space.sample()  # 使用随机策略
#         next_state, reward, done, _, info = env.step(action)
#         trajectory.append((state, action, reward, next_state, done))
#         state = next_state

#     return trajectory  # 直接返回轨迹，而不是使用队列


# class AsyncSampler:
#     def __init__(self, envs, steps_per_env=1024):

#         self.steps_per_env = steps_per_env
#         self.envs = envs
#         self.num_envs = 128

#     def sample(self):
#         # 启动多个异步任务进行采样
#         with multiprocessing.Pool(128) as pool:
#             result = pool.starmap(
#                 sample_env, [(self.steps_per_env, i) for i in range(128)])
#         # tasks = [sample_env.remote(self.envs[i], self.steps_per_env)
#         #          for i in range(self.num_envs)]
#         return result

#     def get_trajectories(self):
#         trajectories = []
#         # 从队列中取出数据，直到队列为空
#         while not self.output_queue.empty():
#             trajectories.append(self.output_queue.get())
#         return trajectories


# def ppo_train(envs_id, num_envs=128, steps_per_env=1024, num_epochs=10):
#     sampler = AsyncSampler("h1hand-walk-v0", steps_per_env)

#     for epoch in range(num_epochs):
#         # Step 1: 异步采样
#         trajectories = sampler.sample()
#         # trajectories = ray.get(tasks)  # 获取所有任务的结果
#         # trajectories = sampler.sample()
#         # Step 2: 打印轨迹的长度或进行其他处理
#         for i, trajectory in enumerate(trajectories):
#             print(f"Env {i}: {len(trajectory)} steps")

#         # Step 3: 网络更新（假设你已经有了 PPO 算法的实现）
#         # 这里调用 PPO 的更新函数进行模型训练，使用 `trajectories` 数据进行更新
#         # Step 4: 重置环境


# if __name__ == "__main__":
#     # envs = [gym.make("h1hand-walk-v0") for _ in range(16)]
#     ppo_train(envs_id="h1hand-walk-v0", num_envs=128,
#               steps_per_env=1024, num_epochs=10)

# def gen_video(self, name="video.mp4"):
#     print('--- rendering video ---')
#     env = gym.make(env_name, render_mode="rgb_array")
#     fps = env.metadata['render_fps']
#     env.reset()
#     with torch.no_grad():
#         done = False
#         video = []
#         while done == False:
#             action = env.action_space.sample()
#             next_state, reward, done, _, info = env.step(action)
#             video.append(env.render())

#     with imageio.get_writer(name, fps=fps) as writer:
#         for frame in video:
#             writer.append_data(frame.astype(np.uint8))
#     env.close()
#     print('--- finish ---')


# def make_env(rank, seed=0):
#     def _init():
#         env = gym.make(env_name, render_mode="rgb_array")
#         env.seed(seed + rank)
#         return env
#     return _init
