import typing

import jax
import jax.numpy as jnp


class Exp3_IX:
    def __init__(self, n_arms: int, learning_rate_scheduler: typing.Generator) -> None:
        self.n_arms = n_arms
        self.sel_probs = jnp.ones(n_arms) / n_arms
        self.lr_sclr = learning_rate_scheduler

    def update(self, selected: int, reward: float, lr: float):
        assert 0 <= reward <= 1
        loss_est = (1-reward) / (self.sel_probs[selected] + lr/2)
        factor = jnp.exp(-lr * loss_est)
        self.sel_probs = self.sel_probs.at[selected].multiply(factor)
        self.sel_probs = self.sel_probs / self.sel_probs.sum()

    def action(self, observations: tuple[int, float]):
        lr = next(self.lr_sclr)
        self.update(*observations, lr)
        return self.sel_probs
    

if __name__ == "__main__":
    def const_value(): 
        while True: yield 0.1

    rng = jax.random.PRNGKey(42)
    experiences = {
        0: 0.9,
        1: 0.1
    }
    action_space = jnp.array(list(experiences.keys()))
    agent = Exp3_IX(2, const_value())
    action = 0
    for i in range(100):
        reward = experiences[action]
        probs = agent.action((action, reward))
        rng, subkey = jax.random.split(rng)
        action = jax.random.choice(subkey, action_space, p=probs).item()
    print(agent.sel_probs)
