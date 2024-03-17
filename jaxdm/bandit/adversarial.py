from typing import Generator, Union

import jax
import jax.numpy as jnp


class Exp3_IX:
    def __init__(self, n_arms: int, learning_rate: Union[float, Generator]) -> None:
        """Exp3-IX algorithm from 
            Algorithm 10 in https://tor-lattimore.com/downloads/book/book.pdf.

        Parameters
        ----------
        n_arms : int
            number of arms.
        learning_rate : Union[float, Generator]
            a fixed learning rate or scheduler.
        """
        self.n_arms = n_arms
        self.sel_probs = jnp.ones(n_arms) / n_arms
        self.learning_rate = learning_rate

    def update(self, selected: int, reward: float, lr: float):
        """One step update using importance weighted estimator.

        Parameters
        ----------
        selected : int
            the selected arm id.
        reward : float
            a reward between [0, 1].
        lr : float
            the learning rate value.
        """
        assert 0 <= reward <= 1
        loss_est = (1-reward) / (self.sel_probs[selected] + lr/2)
        factor = jnp.exp(-lr * loss_est)
        self.sel_probs = self.sel_probs.at[selected].multiply(factor)
        self.sel_probs = self.sel_probs / self.sel_probs.sum()

    def action(self, observations: tuple[int, float]):
        """Return the selection probability on arms.

        Parameters
        ----------
        observations : tuple[int, float]
            (arm_id, reward)

        Returns
        -------
        Array
            a selection probability array
        """
        if isinstance(self.learning_rate, float):
            lr = self.learning_rate
        else:
            lr = next(self.learning_rate)
        self.update(*observations, lr)
        return self.sel_probs
    

if __name__ == "__main__":
    rng = jax.random.PRNGKey(42)
    experiences = {
        0: 0.9,
        1: 0.1
    }
    action_space = jnp.array(list(experiences.keys()))

    from jaxdm.opt.learning_rate import exponent_scheduler
    agent = Exp3_IX(2, learning_rate=exponent_scheduler(0.1))
    action = 0
    for i in range(1000):
        reward = experiences[action]
        probs = agent.action((action, reward))
        rng, subkey = jax.random.split(rng)
        action = jax.random.choice(subkey, action_space, p=probs).item()
    print(agent.sel_probs)
