# Copyright (c) 2024 Boston Dynamics AI Institute LLC. All rights reserved.

# Code modified from https://github.com/amacati/dextgen, MIT license
"""The ``actor`` module contains the actor class as well as the actor networks.

The :class:`.Actor` acts as a wrapper around the actual deterministic policy network to provide
action selection,  and loading utilities.

:class:`.DDP` is a vanilla deep deterministic policy network implementation.
"""
from io import BufferedWriter
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from torch import FloatTensor

from jacta.planner.learning.normalizer import Normalizer
from jacta.planner.learning.replay_buffer import ReplayBuffer


def soft_update(network: nn.Module, target: nn.Module, tau: float) -> nn.Module:
    """Perform a soft update of the target network's weights.

    Shifts the weights of the ``target`` by a factor of ``tau`` into the direction of the
    ``network``.

    Args:
        network: Network from which to copy the weights.
        target: Network that gets updated.
        tau: Controls how much the weights are shifted. Valid in [0, 1].

    Returns:
        The updated target network.
    """
    for network_p, target_p in zip(network.parameters(), target.parameters(), strict=True):
        target_p.data.copy_(tau * network_p.data + (1 - tau) * target_p)
    return target


class Actor:
    """Actor class encapsulating the action selection and training process for the DDPG actor."""

    def __init__(
        self,
        size_s: int,
        size_a: int,
        nlayers: int = 4,
        layer_width: int = 256,
        lr: float = 0.001,
        eps: float = 0.3,
        action_clip: float = 1.0,
    ) -> None:
        """Initialize the actor, the actor networks and create its target as an exact copy.

        Args:
            size_s: Actor network input state size. If the input consists of a state and a
                goal, the size is equal to their sum.
            size_a: Actor network output action size.
            nlayers: Number of network layers.
            layer_width: Number of nodes per layer. Does not influence input and output size.
            lr: Actor network learning rate.
            eps: Random action probability during training.
            action_clip: Action output clipping value.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.action_net = DDP(size_s, size_a, nlayers, layer_width).to(self.device)
        self.target_net = DDP(size_s, size_a, nlayers, layer_width).to(self.device)
        self.target_net.load_state_dict(self.action_net.state_dict())

        self.optim = torch.optim.Adam(self.action_net.parameters(), lr=lr)

        self.eps = eps
        self.action_clip = action_clip
        self._train = False
        self.is_trained = False  # Flag to check if expert has been pretrained

    def select_action(
        self,
        state_norm: Normalizer,
        state: FloatTensor,
        goal: FloatTensor,
        exploration_function: Optional[Callable] = None,
    ) -> FloatTensor:
        """Select an action for the given input states.

        If in train mode, samples noise and chooses completely random actions with probability
        `self.eps`. If in evaluation mode, only clips the action to the maximum value.

        Args:
            states: Input states.

        Returns:
            A numpy array of actions.
        """
        states = state_norm.wrap_obs(state, goal)

        actions = self(states)
        if self._train:
            if torch.rand(1).item() >= self.eps:  # With noise for training
                actions += torch.normal(mean=torch.zeros_like(actions), std=torch.ones_like(actions) * 0.2)
                torch.clip(actions, -self.action_clip, self.action_clip, out=actions)  # In-place op
            elif exploration_function is not None:  # planner exploration
                actions = exploration_function(state)
            else:  # fully random sampling for exploration
                actions = torch.rand(actions.shape) * 2 * self.action_clip - self.action_clip

        return actions

    def __call__(self, states: FloatTensor) -> FloatTensor:
        """Run a forward pass directly on the action net.

        Args:
            states: Input states.

        Returns:
            An action tensor.
        """
        return self.action_net(states)

    def backward_step(self, loss: FloatTensor) -> None:
        """Perform a backward pass with an optimizer step.

        Args:
            loss: Actor network loss.
        """
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def target(self, states: FloatTensor) -> FloatTensor:
        """Compute actions with the target network and without noise.

        Args:
            states: Input states.

        Returns:
            An action tensor.
        """
        return self.target_net(states)

    def eval(self) -> None:
        """Set the actor to eval mode without noise in the action selection."""
        self._train = False
        self.action_net.eval()

    def train(self) -> None:
        """Set the actor to train mode with noisy actions."""
        self._train = True
        self.action_net.train()

    def update_target(self, tau: float = 0.05) -> None:
        """Update the target network with a soft parameter transfer update.

        Args:
            tau: Averaging fraction of the parameter update for the action network.
        """
        soft_update(self.action_net, self.target_net, tau)

    def load(self, checkpoint: Any) -> None:
        """Load data for the actor.

        Args:
            checkpoint: dict containing loaded data.
        """
        self.action_net.load_state_dict(checkpoint["model_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optim.load_state_dict(checkpoint["optimizer_state_dict"])

    def save(self, f: BufferedWriter) -> None:
        """Save data for the actor."""
        torch.save(
            {
                "model_state_dict": self.action_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optim.state_dict(),
            },
            f,
        )


class DDP(nn.Module):
    """Continuous action choice network for the agent."""

    def __init__(self, size_s: int, size_a: int, nlayers: int, layer_width: int) -> None:
        """Initialize the network.

        Args:
            size_s: Input layer size.
            size_a: Output layer size.
            nlayers: Number of network layers.
            layer_width: Number of nodes per layer. Does not influence input and output size.
        """
        assert nlayers >= 1
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(nlayers):
            size_in = size_s if i == 0 else layer_width
            size_out = size_a if i == nlayers - 1 else layer_width
            self.layers.append(nn.Linear(size_in, size_out))
            activation = nn.Tanh() if i == nlayers - 1 else nn.ReLU()
            self.layers.append(activation)

    def forward(self, x: FloatTensor) -> FloatTensor:
        """Compute the network forward pass.

        Args:
            x: Input tensor.

        Returns:
            The network output.
        """
        for layer in self.layers:
            x = layer(x)
        return x


"""The ``critic`` module contains the critic class as well as the critic network.

The :class:`.Critic` acts as a wrapper around the actual critic Q-function to provide loading utilities.

:class:`.CriticNetwork` is a vanilla deep state-action network implementation.
"""


class Critic:
    """Critic class encapsulating the critic and training process for the DDPG critic."""

    def __init__(self, size_s: int, size_a: int, nlayers: int = 4, layer_width: int = 256, lr: float = 0.001) -> None:
        """Initialize the critic, the critic network and create its target as an exact copy.

        Args:
            size_s: Critic network input state size. If the input consists of a state and a
                goal, the size is equal to their sum.
            size_a: Critic network input action size.
            nlayers: Number of network layers.
            layer_width: Number of nodes per layer. Does not influence input and output size.
            lr: Critic network learning rate.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.critic_net = CriticNetwork(size_s, size_a, nlayers, layer_width).to(self.device)
        self.target_net = CriticNetwork(size_s, size_a, nlayers, layer_width).to(self.device)
        self.target_net.load_state_dict(self.critic_net.state_dict())

        self.optim = torch.optim.Adam(self.critic_net.parameters(), lr=lr)

    def __call__(self, states: FloatTensor, actions: FloatTensor) -> FloatTensor:
        """Run a critic net forward pass.

        Args:
            states: Input states.
            actions: Input actions.

        Returns:
            An action value tensor.
        """
        return self.critic_net(states, actions)

    def target(self, states: FloatTensor, actions: FloatTensor) -> FloatTensor:
        """Compute the action value with the target network.

        Args:
            states: Input states.
            actions: Input actions.

        Returns:
            An action value tensor.
        """
        return self.target_net(states, actions)

    def backward_step(self, loss: FloatTensor) -> None:
        """Perform a backward pass with an optimizer step.

        Args:
            loss: Critic network loss.
        """
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def update_target(self, tau: float = 0.05) -> None:
        """Update the target network with a soft parameter transfer update.

        Args:
            tau: Averaging fraction of the parameter update for the action network.
        """
        soft_update(self.critic_net, self.target_net, tau)

    def load(self, checkpoint: Any) -> None:
        """Load data for the critic.

        Args:
            checkpoint: dict containing loaded data.
        """
        self.critic_net.load_state_dict(checkpoint["model_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        self.optim.load_state_dict(checkpoint["optimizer_state_dict"])

    def save(self, f: BufferedWriter) -> None:
        """Save data for the critic."""
        torch.save(
            {
                "model_state_dict": self.critic_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optim.state_dict(),
            },
            f,
        )


class CriticNetwork(nn.Module):
    """State action critic network for the critic."""

    def __init__(self, size_s: int, size_a: int, nlayers: int, layer_width: int) -> None:
        """Initialize the network.

        Args:
            size_s: Input layer size.
            size_a: Output layer size.
            nlayers: Number of network layers.
            layer_width: Number of nodes per layer. Does not influence input and output size.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(nlayers):
            size_in = size_s + size_a if i == 0 else layer_width
            size_out = 1 if i == nlayers - 1 else layer_width
            self.layers.append(nn.Linear(size_in, size_out))
            if i != nlayers - 1:  # Last layer doesn't get an activation function
                self.layers.append(nn.ReLU())

    def forward(self, state: FloatTensor, action: FloatTensor) -> FloatTensor:
        """Compute the network forward pass.

        Args:
            state: Input state tensor.
            action: Input action tensor.

        Returns:
            The network output.
        """
        x = torch.cat([state, action], dim=1)
        for layer in self.layers:
            x = layer(x)
        return x


def train_actor_critic(
    actor: Actor,
    actor_expert: Actor,
    critic: Critic,
    state_norm: Normalizer,
    replay_buffer: ReplayBuffer,
    reward_fun: Callable,
    batch_size: int = 256,
    discount_factor: float = 0.98,
    her_probability: float = 0.0,
) -> float:
    """Train the agent and critic network with experience sampled from the replay buffer."""
    states, actions, rewards, next_states, goals, _, _, _ = replay_buffer.sampling(
        batch_size, her_probability, reward_fun
    )
    obs = state_norm.wrap_obs(states, goals)
    obs_next = state_norm.wrap_obs(next_states, goals)
    with torch.no_grad():
        next_actions = actor.target(obs_next)
        next_q = critic.target(obs_next, next_actions)
        value = rewards + discount_factor * next_q
        # Clip to minimum reward possible, geometric sum over the finite horizon with discount_factor and -1 rewards
        min_value = -1 * (1 - discount_factor**replay_buffer.params.learner_trajectory_length) / (1 - discount_factor)
        torch.clip(value, min_value, 0, out=value)

    if actor_expert.is_trained:
        with torch.no_grad():
            next_actions_expert = actor_expert.target(obs_next)
            next_q_expert = critic.target(obs_next, next_actions_expert)
            value_expert = rewards + discount_factor * next_q_expert
            torch.clip(value_expert, min_value, 0, out=value_expert)
            expert_share = torch.sum(value_expert > value) / batch_size
            value = torch.max(value_expert, value)  # use highest possible value as reference for Q
    else:
        expert_share = 0.0

    q = critic(obs, actions)
    critic_loss = (value - q).pow(2).mean()

    actions = actor(obs)
    next_q = critic(obs, actions)
    actor_loss = -next_q.mean()
    actor.backward_step(actor_loss)
    critic.backward_step(critic_loss)

    return expert_share


def train_actor_imitation(
    actor: Actor,
    state_norm: Normalizer,
    states: FloatTensor,
    goals: FloatTensor,
    actor_actions: FloatTensor,
    batch_size: int = 256,
) -> float:
    indices = torch.randint(0, len(states), (batch_size,))
    obs = state_norm.wrap_obs(states[indices], goals[indices])
    actor_actions_learner = actor(obs)
    actor_actions_planner = actor_actions[indices]
    loss = (actor_actions_learner - actor_actions_planner).pow(2).mean()
    actor.backward_step(loss)

    actor.target_net.load_state_dict(actor.action_net.state_dict())

    return loss


def train_critic_imitation(
    critic: Critic,
    state_norm: Normalizer,
    states: FloatTensor,
    goals: FloatTensor,
    actor_actions: FloatTensor,
    q_values: FloatTensor,
    batch_size: int = 256,
) -> float:
    indices = torch.randint(0, len(states), (batch_size,))
    obs = state_norm.wrap_obs(states[indices], goals[indices])
    actor_actions = actor_actions[indices]
    q_values_learner = critic(obs, actor_actions)
    q_values_planner = q_values[indices]
    loss = (q_values_learner - q_values_planner).pow(2).mean()
    critic.backward_step(loss)

    critic.target_net.load_state_dict(critic.critic_net.state_dict())

    return loss
