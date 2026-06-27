from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch


@dataclass(slots=True)
class SequenceBatch:
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor


@dataclass(slots=True)
class SubsequenceBatch:
    o_prev: torch.Tensor
    o_t: torch.Tensor
    o_next: torch.Tensor
    a_t: torch.Tensor
    r_t: torch.Tensor
    done_t: torch.Tensor
    sequence_index: torch.Tensor
    time_index: torch.Tensor
    observations_seq: torch.Tensor
    actions_seq: torch.Tensor
    rewards_seq: torch.Tensor
    dones_seq: torch.Tensor


class SequenceReplayBuffer:
    def __init__(self, capacity: int, sequence_length: int) -> None:
        if sequence_length < 3:
            raise ValueError("sequence_length must be >= 3")
        self.capacity = capacity
        self.sequence_length = sequence_length
        self._storage: deque[SequenceBatch] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._storage)

    def add(self, batch: SequenceBatch) -> None:
        if batch.observations.shape[0] < self.sequence_length:
            raise ValueError("Sequence shorter than configured sequence_length.")
        self._storage.append(batch)

    def add_trajectory(self, observations: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor) -> int:
        if not (len(observations) == len(actions) == len(rewards) == len(dones)):
            raise ValueError("Trajectory tensors must have equal temporal length.")
        if len(observations) < self.sequence_length:
            raise ValueError("Trajectory shorter than configured sequence_length.")

        inserted = 0
        for start in range(0, len(observations) - self.sequence_length + 1):
            end = start + self.sequence_length
            self.add(
                SequenceBatch(
                    observations=observations[start:end].clone(),
                    actions=actions[start:end].clone(),
                    rewards=rewards[start:end].clone(),
                    dones=dones[start:end].clone(),
                )
            )
            inserted += 1
        return inserted

    def sample(self, batch_size: int) -> SequenceBatch:
        if batch_size > len(self._storage):
            raise ValueError("Not enough sequences in replay buffer.")
        indices = torch.randint(0, len(self._storage), (batch_size,))
        items = [self._storage[int(idx)] for idx in indices]
        return SequenceBatch(
            observations=torch.stack([item.observations for item in items], dim=0),
            actions=torch.stack([item.actions for item in items], dim=0),
            rewards=torch.stack([item.rewards for item in items], dim=0),
            dones=torch.stack([item.dones for item in items], dim=0),
        )

    def sample_subsequences(self, batch_size: int) -> SubsequenceBatch:
        sequences = self.sample(batch_size)
        o_prev = []
        o_t = []
        o_next = []
        a_t = []
        r_t = []
        done_t = []
        sequence_index = []
        time_index = []

        for seq_idx in range(sequences.observations.shape[0]):
            seq_obs = sequences.observations[seq_idx]
            seq_actions = sequences.actions[seq_idx]
            seq_rewards = sequences.rewards[seq_idx]
            seq_dones = sequences.dones[seq_idx]
            length = seq_obs.shape[0]
            
            for t in range(1, length - 1):
                o_prev.append(seq_obs[t - 1])
                o_t.append(seq_obs[t])
                o_next.append(seq_obs[t + 1])
                a_t.append(seq_actions[t])
                r_t.append(seq_rewards[t])
                done_t.append(seq_dones[t])
                sequence_index.append(seq_idx)
                time_index.append(t)

        return SubsequenceBatch(
            o_prev=torch.stack(o_prev, dim=0),
            o_t=torch.stack(o_t, dim=0),
            o_next=torch.stack(o_next, dim=0),
            a_t=torch.stack(a_t, dim=0),
            r_t=torch.stack(r_t, dim=0),
            done_t=torch.stack(done_t, dim=0),
            sequence_index=torch.tensor(sequence_index, dtype=torch.long),
            time_index=torch.tensor(time_index, dtype=torch.long),
            observations_seq=torch.stack([sequences.observations[seq_idx] for seq_idx in sequence_index], dim=0),
            actions_seq=torch.stack([sequences.actions[seq_idx] for seq_idx in sequence_index], dim=0),
            rewards_seq=torch.stack([sequences.rewards[seq_idx] for seq_idx in sequence_index], dim=0),
            dones_seq=torch.stack([sequences.dones[seq_idx] for seq_idx in sequence_index], dim=0),
        )
