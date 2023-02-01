# Will contain registration code for environment name and where to find the environment file
from gym.envs.registration import register

register(id='snake-v0', entry_point="snake.envs:SnakeEnv")