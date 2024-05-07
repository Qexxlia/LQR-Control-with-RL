import numpy as np

from stable_baselines3.common.callbacks import BaseCallback

class TensorBoardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorBoardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        reward = self.training_env.get_attr("reward")[0]
        self.logger.record("reward", reward)
        return True