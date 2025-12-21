import cProfile
import random

from rl.env import TractorEnv


def step_randomly(n: int = 100):
    env = TractorEnv()
    obs, _ = env.reset()
    for _ in range(n):
        actions = {}
        for agent, agent_obs in obs.items():
            choices = [
                i
                for i in range(agent_obs["action_mask"].size)
                if agent_obs["action_mask"][i] == 1.0
            ]
            actions[agent] = random.choice(choices)
        obs, _, _, _, _ = env.step(actions)
        if env.game.game_over():
            obs, _ = env.reset()


cProfile.run("step_randomly(100000)", sort="cumtime")
