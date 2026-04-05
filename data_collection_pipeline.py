import numpy as np
import pickle
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple
from pathlib import Path


class CommunicationTracer:
    """Captures and logs inter-agent communication patterns"""

    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        self.reset()

    def reset(self):
        self.traces = []
        self.timestep = 0

    def log_step(self, observations: Dict, actions: Dict, rewards: Dict,
                 infos: Dict = None):
        step_data = {
            'timestep': self.timestep,
            'observations': {agent: obs.tolist() if hasattr(obs, 'tolist') else obs
                             for agent, obs in observations.items()},
            'actions': {agent: int(action) if hasattr(action, 'item') else action
                        for agent, action in actions.items()},
            'rewards': {agent: float(reward) for agent, reward in rewards.items()},
            'infos': infos if infos else {}
        }
        self.traces.append(step_data)
        self.timestep += 1

    def get_communication_features(self) -> Dict:
        if not self.traces:
            return {}
        features = {
            'episode_length': len(self.traces),
            'total_reward': sum(sum(step['rewards'].values()) for step in self.traces),
            'avg_reward': np.mean([sum(step['rewards'].values()) for step in self.traces]),
            'coordination_events': self._count_coordination_events(),
            'action_diversity': self._measure_action_diversity(),
            'reward_variance': np.var([sum(step['rewards'].values()) for step in self.traces])
        }
        return features

    def _count_coordination_events(self) -> int:
        coord_count = 0
        for step in self.traces:
            actions = list(step['actions'].values())
            if len(set(actions)) < len(actions) / 2:
                coord_count += 1
        return coord_count

    def _measure_action_diversity(self) -> float:
        all_actions = []
        for step in self.traces:
            all_actions.extend(step['actions'].values())
        if not all_actions:
            return 0.0
        return len(set(all_actions)) / len(all_actions)

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump(self.traces, f)


class FailureInjector:
    """Injects synthetic coordination failures into agent communication"""

    FAILURE_TYPES = [
        'communication_delay',
        'message_drop',
        'observation_noise',
        'action_corruption',
        'reward_misalignment',
        'observation_freeze',
        'observation_bias',
        'observation_sign_flip',
        'partial_obs_drop',
        'action_delay',
        'action_repeat',
        'action_stuck_0',
        'action_stuck_1',
        'reward_freeze',
        'reward_delay',
        'reward_zero',
        'reward_constant',
        'extreme_obs_noise'
    ]

    def __init__(self, failure_type: str, severity: float = 0.3):
        assert failure_type in self.FAILURE_TYPES, f"Unknown failure type: {failure_type}"
        self.failure_type = failure_type
        # FIX #5: randomise severity per-episode so effect strength varies
        # across episodes, preventing each class from having a unique fixed
        # signal magnitude. Clipped to [0.1, 0.9] to stay meaningful.
        self.severity = float(np.clip(
            severity * np.random.uniform(0.5, 1.5), 0.1, 0.9
        ))
        self.base_severity = severity
        self.delay_buffer = []
        self.action_delay_buffer = []
        self.reward_delay_buffer = []
        self.frozen_obs = None
        self.frozen_reward = None
        self.last_action = None

        # FIX #5: per-episode random parameters sampled once at construction
        # so they are consistent within an episode but vary across episodes.
        self._obs_bias_offset = float(np.random.normal(
            10.0 * self.severity, 2.0 * self.severity
        ))
        self._extreme_noise_scale = float(
            np.random.uniform(3.0, 7.0) * self.severity
        )
        self._reward_constant_val = float(np.random.uniform(80.0, 120.0))
        self._reward_zero_prob = float(np.clip(
            self.severity + np.random.uniform(-0.1, 0.1), 0.05, 0.95
        ))
        self._sign_flip_prob = float(np.clip(
            self.severity + np.random.uniform(-0.1, 0.1), 0.05, 0.95
        ))

        # FIX #5: background noise added to ALL episodes (obs + reward)
        # Small enough not to overwhelm the signal, large enough to create
        # realistic overlap between classes in feature space.
        self._bg_obs_noise_std   = 0.05
        self._bg_reward_noise_std = 0.02

    def inject(self, observations: Dict, actions: Dict = None,
               rewards: Dict = None, timestep: int = 0) -> Tuple[Dict, Dict, Dict]:
        """
        Inject failure into observations/actions/rewards in a single call.
        FIX #1: called ONCE per step to avoid stateful buffer corruption.
        """
        if self.failure_type == 'communication_delay':
            return self._inject_delay(observations, timestep), actions, rewards
        elif self.failure_type == 'message_drop':
            return self._inject_drop(observations), actions, rewards
        elif self.failure_type == 'observation_noise':
            return self._inject_noise(observations), actions, rewards
        elif self.failure_type == 'action_corruption':
            return observations, self._corrupt_actions(actions), rewards
        elif self.failure_type == 'reward_misalignment':
            return observations, actions, self._misalign_rewards(rewards)
        elif self.failure_type == 'observation_freeze':
            return self._inject_obs_freeze(observations), actions, rewards
        elif self.failure_type == 'observation_bias':
            return self._inject_obs_bias(observations), actions, rewards
        elif self.failure_type == 'observation_sign_flip':
            return self._inject_obs_sign_flip(observations), actions, rewards
        elif self.failure_type == 'partial_obs_drop':
            return self._inject_partial_obs_drop(observations), actions, rewards
        elif self.failure_type == 'action_delay':
            return observations, self._inject_action_delay(actions), rewards
        elif self.failure_type == 'action_repeat':
            return observations, self._inject_action_repeat(actions), rewards
        elif self.failure_type == 'action_stuck_0':
            return observations, self._inject_action_stuck(actions, 0), rewards
        elif self.failure_type == 'action_stuck_1':
            return observations, self._inject_action_stuck(actions, 1), rewards
        elif self.failure_type == 'reward_freeze':
            return observations, actions, self._inject_reward_freeze(rewards)
        elif self.failure_type == 'reward_delay':
            return observations, actions, self._inject_reward_delay(rewards)
        elif self.failure_type == 'reward_zero':
            return observations, actions, self._inject_reward_zero(rewards)
        elif self.failure_type == 'reward_constant':
            return observations, actions, self._inject_reward_constant(rewards)
        elif self.failure_type == 'extreme_obs_noise':
            return self._inject_extreme_obs_noise(observations), actions, rewards
        return observations, actions, rewards

    # ── helpers ────────────────────────────────────────────────────────

    def _add_bg_obs_noise(self, observations: Dict) -> Dict:
        """FIX #5: add small background noise to all obs to create class overlap."""
        return {
            agent: obs + np.random.normal(0, self._bg_obs_noise_std, size=obs.shape)
            for agent, obs in observations.items()
        }

    def _add_bg_reward_noise(self, rewards: Dict) -> Dict:
        """FIX #5: add small background noise to all rewards."""
        if rewards is None:
            return rewards
        return {
            agent: reward + np.random.normal(0, self._bg_reward_noise_std)
            for agent, reward in rewards.items()
        }

    # ── observation injectors ──────────────────────────────────────────

    def _inject_delay(self, observations: Dict, timestep: int) -> Dict:
        delay_steps = max(1, int(3 * self.severity))
        self.delay_buffer.append(observations.copy())
        if len(self.delay_buffer) > delay_steps:
            obs = self.delay_buffer.pop(0)
        else:
            obs = observations
        return self._add_bg_obs_noise(obs)

    def _inject_drop(self, observations: Dict) -> Dict:
        modified_obs = {}
        for agent, obs in observations.items():
            if np.random.random() < self.severity:
                modified_obs[agent] = np.zeros_like(obs)
            else:
                modified_obs[agent] = obs
        return self._add_bg_obs_noise(modified_obs)

    def _inject_noise(self, observations: Dict) -> Dict:
        modified_obs = {}
        for agent, obs in observations.items():
            noise = np.random.normal(0, self.severity * 0.5, size=obs.shape)
            modified_obs[agent] = obs + noise
        return self._add_bg_obs_noise(modified_obs)

    def _inject_obs_freeze(self, observations: Dict) -> Dict:
        if self.frozen_obs is None or np.random.random() < (1.0 - self.severity):
            self.frozen_obs = observations.copy()
        return self._add_bg_obs_noise(self.frozen_obs)

    def _inject_obs_bias(self, observations: Dict) -> Dict:
        # FIX #5: use per-episode sampled bias instead of fixed 10*severity
        result = {agent: obs + self._obs_bias_offset
                  for agent, obs in observations.items()}
        return self._add_bg_obs_noise(result)

    def _inject_obs_sign_flip(self, observations: Dict) -> Dict:
        # FIX #5: use per-episode sampled flip probability
        result = {
            agent: -obs if np.random.random() < self._sign_flip_prob else obs
            for agent, obs in observations.items()
        }
        return self._add_bg_obs_noise(result)

    def _inject_partial_obs_drop(self, observations: Dict) -> Dict:
        modified_obs = {}
        for agent, obs in observations.items():
            mod_obs = obs.copy()
            mask = np.random.random(mod_obs.shape) < self.severity
            mod_obs[mask] = 0
            modified_obs[agent] = mod_obs
        return self._add_bg_obs_noise(modified_obs)

    def _inject_extreme_obs_noise(self, observations: Dict) -> Dict:
        # FIX #5: use per-episode sampled noise scale instead of fixed 5*severity
        result = {
            agent: obs + np.random.normal(0, self._extreme_noise_scale, size=obs.shape)
            for agent, obs in observations.items()
        }
        return self._add_bg_obs_noise(result)

    # ── action injectors ───────────────────────────────────────────────

    def _corrupt_actions(self, actions: Dict) -> Dict:
        if actions is None:
            return None
        return {
            agent: np.random.randint(0, 5) if np.random.random() < self.severity
            else action
            for agent, action in actions.items()
        }

    def _inject_action_delay(self, actions: Dict) -> Dict:
        if actions is None:
            return None
        delay_steps = max(1, int(3 * self.severity))
        self.action_delay_buffer.append(actions.copy())
        if len(self.action_delay_buffer) > delay_steps:
            return self.action_delay_buffer.pop(0)
        return self.action_delay_buffer[0]

    def _inject_action_repeat(self, actions: Dict) -> Dict:
        if actions is None:
            return None
        if self.last_action is None or np.random.random() > self.severity:
            self.last_action = actions.copy()
        return self.last_action

    def _inject_action_stuck(self, actions: Dict, val: int) -> Dict:
        if actions is None:
            return None
        return {
            agent: val if np.random.random() < self.severity else action
            for agent, action in actions.items()
        }

    # ── reward injectors ───────────────────────────────────────────────

    def _misalign_rewards(self, rewards: Dict) -> Dict:
        if rewards is None:
            return None
        modified = {}
        for agent, reward in rewards.items():
            if np.random.random() < self.severity:
                modified[agent] = (-reward if np.random.random() > 0.5
                                   else reward + np.random.normal(0, 0.5))
            else:
                modified[agent] = reward
        return self._add_bg_reward_noise(modified)

    def _inject_reward_freeze(self, rewards: Dict) -> Dict:
        if rewards is None:
            return None
        if self.frozen_reward is None or np.random.random() > self.severity:
            self.frozen_reward = rewards.copy()
        return self._add_bg_reward_noise(self.frozen_reward)

    def _inject_reward_delay(self, rewards: Dict) -> Dict:
        if rewards is None:
            return None
        delay_steps = max(1, int(3 * self.severity))
        self.reward_delay_buffer.append(rewards.copy())
        if len(self.reward_delay_buffer) > delay_steps:
            delayed = self.reward_delay_buffer.pop(0)
        else:
            delayed = self.reward_delay_buffer[0]
        return self._add_bg_reward_noise(delayed)

    def _inject_reward_zero(self, rewards: Dict) -> Dict:
        if rewards is None:
            return None
        # FIX #5: use per-episode sampled zero probability
        result = {
            agent: 0.0 if np.random.random() < self._reward_zero_prob else reward
            for agent, reward in rewards.items()
        }
        return self._add_bg_reward_noise(result)

    def _inject_reward_constant(self, rewards: Dict) -> Dict:
        if rewards is None:
            return None
        # FIX #5: use per-episode sampled constant value instead of fixed 100.0
        result = {
            agent: self._reward_constant_val
                   if np.random.random() < self.severity else reward
            for agent, reward in rewards.items()
        }
        return self._add_bg_reward_noise(result)

    def reset(self):
        self.delay_buffer = []
        self.action_delay_buffer = []
        self.reward_delay_buffer = []
        self.frozen_obs = None
        self.frozen_reward = None
        self.last_action = None


class DataCollector:
    """Main data collection orchestrator"""

    def __init__(self, output_dir: str = "./marl_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        (self.output_dir / "normal").mkdir(exist_ok=True)
        (self.output_dir / "failures").mkdir(exist_ok=True)
        for failure_type in FailureInjector.FAILURE_TYPES:
            (self.output_dir / "failures" / failure_type).mkdir(exist_ok=True)

        self.metadata = {
            'collection_date': datetime.now().isoformat(),
            'episodes_collected': 0,
            'normal_episodes': 0,
            'failure_episodes': {}
        }

    def collect_normal_episodes(self, env, num_episodes: int = 100,
                                policy_fn=None, verbose: bool = True):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Collecting {num_episodes} NORMAL coordination episodes")
            print(f"{'='*60}\n")

        collected = 0
        for episode in range(num_episodes):
            episode_path = self.output_dir / "normal" / f"episode_{episode:04d}.pkl"
            if episode_path.exists():
                if verbose and episode == 0:
                    print("  Resuming — skipping already-collected episodes...")
                continue

            observations, infos = env.reset()
            num_agents = len(env.agents)
            tracer = CommunicationTracer(num_agents)

            done = False
            step_count = 0

            while not done:
                if policy_fn is None:
                    actions = {agent: env.action_space(agent).sample()
                               for agent in env.agents}
                else:
                    actions = {agent: policy_fn(observations[agent], agent)
                               for agent in env.agents}

                next_observations, rewards, terminations, truncations, infos = env.step(actions)
                tracer.log_step(observations, actions, rewards, infos)
                observations = next_observations
                step_count += 1
                done = all(terminations.values()) or all(truncations.values())

            tracer.save(episode_path)
            features = tracer.get_communication_features()
            self._save_episode_metadata(episode, "normal", features,
                                        first_episode=(episode == 0))
            collected += 1

            if verbose and (episode + 1) % 10 == 0:
                print(f"  Collected {episode + 1}/{num_episodes} episodes | "
                      f"Last episode: {step_count} steps, "
                      f"Reward: {features['total_reward']:.2f}")

        self.metadata['normal_episodes'] = num_episodes
        if verbose:
            skipped = num_episodes - collected
            if skipped:
                print(f"  (Skipped {skipped} already-existing episodes)")
            print(f"\n✓ Normal episode collection complete!\n")

    def collect_failure_episodes(self, env, failure_type: str,
                                 num_episodes: int = 50,
                                 severity: float = 0.3,
                                 policy_fn=None,
                                 verbose: bool = True):
        """
        Collect episodes with injected coordination failures.
        FIX #1: single inject() call per step.
        FIX #5: FailureInjector now randomises severity per-episode.
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Collecting {num_episodes} episodes with {failure_type.upper()}")
            print(f"Base severity: {severity}")
            print(f"{'='*60}\n")

        collected = 0
        for episode in range(num_episodes):
            episode_path = (self.output_dir / "failures" / failure_type /
                            f"episode_{episode:04d}.pkl")

            if episode_path.exists():
                if verbose and episode == 0:
                    print("  Resuming — skipping already-collected episodes...")
                continue

            observations, infos = env.reset()
            # FIX #5: new injector per episode → fresh per-episode random params
            injector = FailureInjector(failure_type, severity)
            num_agents = len(env.agents)
            tracer = CommunicationTracer(num_agents)

            done = False
            step_count = 0

            while not done:
                # Inject obs failure
                modified_obs, modified_actions, modified_rewards = injector.inject(
                    observations, actions=None, rewards=None, timestep=step_count
                )

                if policy_fn is None:
                    raw_actions = {agent: env.action_space(agent).sample()
                                   for agent in env.agents}
                else:
                    raw_actions = {agent: policy_fn(modified_obs[agent], agent)
                                   for agent in env.agents}

                # Inject action failure
                _, modified_actions, _ = injector.inject(
                    modified_obs, raw_actions, rewards=None, timestep=step_count
                )

                effective_actions = modified_actions if modified_actions is not None else raw_actions
                next_observations, real_rewards, terminations, truncations, infos = env.step(
                    effective_actions
                )

                # Inject reward failure
                _, _, perceived_rewards = injector.inject(
                    modified_obs, effective_actions, real_rewards, timestep=step_count
                )

                tracer.log_step(
                    modified_obs,
                    effective_actions,
                    perceived_rewards if perceived_rewards is not None else real_rewards,
                    infos
                )

                observations = next_observations
                step_count += 1
                done = all(terminations.values()) or all(truncations.values())

            tracer.save(episode_path)
            features = tracer.get_communication_features()
            features['failure_type'] = failure_type
            features['failure_severity'] = severity
            self._save_episode_metadata(episode, f"failures/{failure_type}", features,
                                        first_episode=(episode == 0))
            collected += 1

            if verbose and (episode + 1) % 10 == 0:
                print(f"  Collected {episode + 1}/{num_episodes} episodes | "
                      f"Last episode: {step_count} steps, "
                      f"Reward: {features['total_reward']:.2f}")

        if failure_type not in self.metadata['failure_episodes']:
            self.metadata['failure_episodes'][failure_type] = 0
        self.metadata['failure_episodes'][failure_type] += num_episodes

        if verbose:
            skipped = num_episodes - collected
            if skipped:
                print(f"  (Skipped {skipped} already-existing episodes)")
            print(f"\n✓ {failure_type} episode collection complete!\n")

    def _save_episode_metadata(self, episode_id: int, category: str,
                               features: Dict, first_episode: bool = False):
        metadata_file = self.output_dir / f"{category.replace('/', '_')}_metadata.json"

        if first_episode or not metadata_file.exists():
            metadata = {'episodes': []}
        else:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

        features['episode_id'] = episode_id
        metadata['episodes'].append(features)

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def save_collection_summary(self):
        summary_file = self.output_dir / "collection_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        print(f"\n{'='*60}")
        print("DATA COLLECTION SUMMARY")
        print(f"{'='*60}")
        print(f"Output directory: {self.output_dir}")
        print(f"Normal episodes: {self.metadata['normal_episodes']}")
        print(f"Failure episodes:")
        for failure_type, count in self.metadata['failure_episodes'].items():
            print(f"  - {failure_type}: {count}")
        print(f"{'='*60}\n")


def create_simple_policy():
    """
    Simple heuristic policy for simple_spread environment.
    FIX #2: exact index parsing instead of string-contains.
    """
    def policy(observation, agent_id):
        if np.random.random() < 0.15:
            return np.random.randint(0, 5)

        l1_rel = observation[4:6]
        l2_rel = observation[6:8]
        l3_rel = observation[8:10]

        try:
            agent_idx = int(str(agent_id).split('_')[-1])
        except (ValueError, IndexError):
            agent_idx = 0

        targets = [l1_rel, l2_rel, l3_rel]
        target = targets[agent_idx % 3]

        dx, dy = target
        if abs(dx) > abs(dy):
            return 1 if dx < 0 else 2
        else:
            return 3 if dy < 0 else 4

    return policy


if __name__ == "__main__":
    print("Data Collection Pipeline Ready!")
    print("\nIMPORTANT: Delete existing marl_dataset/ before re-collecting")
    print("so the new randomised injectors take effect on fresh episodes.")
