from agent_prm.critics.critic import Critic
import random

class RandomCritic(Critic):
    def name(self) -> str:
        return "random"

    def score_reason_action_batch(self, queries):
        scores = [random.random() for _ in range(len(queries))]
        return scores