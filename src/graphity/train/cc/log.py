import pickle
import zipfile

import numpy as np
import pandas as pd
import torch
#######################
#       Logging       #
#######################
class DirLogger:
    def __init__(self, H, log_file=None):
        self.log_file = log_file
        self.H = H
        if log_file:
            self.log_file = zipfile.ZipFile(log_file, "w")

    def close(self):
        if self.log_file: self.log_file.close()

    def mark_corrupt(self):
        if self.log_file:
            with self.log_file.open("corrupt.txt", "w") as file:
                file.touch()

    def log_metainfo(self, metainfo):
        if self.log_file:
            with self.log_file.open("config.pkl", "w") as file:
                pickle.dump(metainfo, file)
            with self.log_file.open("models.pkl", "w") as file:
                models = {idx:alg for idx, alg in enumerate(metainfo['alg'])}
                pickle.dump(models, file)
    def log_main(self, file):
        # Add this file to the zip, so that the zip can be executed.
        if self.log_file:
            self.log_file.write(file, "__main__.py")

    def log_seed(self, epoch, seed):
        if self.log_file:
            with self.log_file.open(f"seed/{epoch:04d}.csv", "w") as file:
                pd.DataFrame(seed.detach().cpu().numpy()).to_csv(file, index=None, header=None)
                
    
    def log_task(self, epochs, task):
        energy_list = []
        for trajectory_idx, trajectory in enumerate(task.trajectories):
            rewards = [trajectory.reward_buffer[idx]for idx in range(min(trajectory.done, len(trajectory.reward_buffer)))]
            rewards = torch.stack(rewards)
            mindex = torch.argmin(rewards)
            energy_list.append(np.exp(trajectory.reward_buffer[mindex].item()))

        if self.log_file:
            for trajectory_idx, trajectory in enumerate(task.trajectories):
                with self.log_file.open(f"models/{task.number:02d}/{epochs:04d}/traj{trajectory_idx}.pkl", "w") as fptr:
                    pickle.dump(trajectory, fptr)

        rewards = len(task.trajectories) * [None]
        for idx, traj in enumerate(task.trajectories): rewards[idx] = sum(traj.reward_buffer)
        print(f"R^bar_({epochs:04d})_{task.name} = {(sum(rewards)/len(rewards)).item():07f}. Best was {round(min(energy_list))}.")