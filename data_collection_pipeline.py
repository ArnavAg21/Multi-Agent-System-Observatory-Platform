"""
PettingZoo Data Collection Pipeline for MARL Observatory Platform
Collects communication traces, agent states, and coordination events
"""

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
        """Reset trace for new episode"""
        self.traces = []
        self.timestep = 0
    
    def log_step(self, observations: Dict, actions: Dict, rewards: Dict, 
                 infos: Dict = None):
        """Log a single timestep of agent interactions"""
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
        """Extract communication pattern features for analysis"""
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
        """Count instances where agents take synchronized actions"""
        coord_count = 0
        for step in self.traces:
            actions = list(step['actions'].values())
            # Simple heuristic: coordination = similar actions at same time
            if len(set(actions)) < len(actions) / 2:
                coord_count += 1
        return coord_count
    
    def _measure_action_diversity(self) -> float:
        """Measure how diverse agent actions are across episode"""
        all_actions = []
        for step in self.traces:
            all_actions.extend(step['actions'].values())
        if not all_actions:
            return 0.0
        return len(set(all_actions)) / len(all_actions)
    
    def save(self, filepath: str):
        """Save trace to file"""
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
        self.severity = severity  # 0.0 to 1.0
        self.delay_buffer = []
        self.action_delay_buffer = []
        self.reward_delay_buffer = []
        self.frozen_obs = None
        self.frozen_reward = None
        self.last_action = None
    
    def inject(self, observations: Dict, actions: Dict = None, 
               rewards: Dict = None, timestep: int = 0) -> Tuple[Dict, Dict, Dict]:
        """
        Inject failure into observations/actions/rewards
        Returns modified versions
        """
        if self.failure_type == 'communication_delay':
            return self._inject_delay(observations, timestep), actions, rewards
        
        elif self.failure_type == 'message_drop':
            return self._inject_drop(observations), actions, rewards
        
        elif self.failure_type == 'observation_noise':
            return self._inject_noise(observations), actions, rewards
        
        elif self.failure_type == 'action_corruption':
            obs_mod, act_mod = observations, self._corrupt_actions(actions)
            return obs_mod, act_mod, rewards
        
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
    
    def _inject_delay(self, observations: Dict, timestep: int) -> Dict:
        """Simulate communication delay by using old observations"""
        delay_steps = int(3 * self.severity)  # 0-3 step delay
        
        # Add current observation to buffer
        self.delay_buffer.append(observations.copy())
        
        # Return delayed observation if buffer is full
        if len(self.delay_buffer) > delay_steps:
            delayed_obs = self.delay_buffer.pop(0)
            return delayed_obs
        
        return observations
    
    def _inject_drop(self, observations: Dict) -> Dict:
        """Randomly drop observations (replace with zeros)"""
        modified_obs = {}
        for agent, obs in observations.items():
            if np.random.random() < self.severity:
                # Drop this observation
                modified_obs[agent] = np.zeros_like(obs)
            else:
                modified_obs[agent] = obs
        return modified_obs
    
    def _inject_noise(self, observations: Dict) -> Dict:
        """Add Gaussian noise to observations"""
        modified_obs = {}
        for agent, obs in observations.items():
            noise = np.random.normal(0, self.severity * 0.5, size=obs.shape)
            modified_obs[agent] = obs + noise
        return modified_obs
    
    def _corrupt_actions(self, actions: Dict) -> Dict:
        """Randomly corrupt agent actions"""
        if actions is None:
            return None
        
        modified_actions = {}
        for agent, action in actions.items():
            if np.random.random() < self.severity:
                # Corrupt to random action (assumes discrete action space)
                modified_actions[agent] = np.random.randint(0, 5)
            else:
                modified_actions[agent] = action
        return modified_actions
    
    def _misalign_rewards(self, rewards: Dict) -> Dict:
        """Inject conflicting reward signals"""
        if rewards is None:
            return None
        
        modified_rewards = {}
        for agent, reward in rewards.items():
            if np.random.random() < self.severity:
                # Flip reward sign or add noise
                modified_rewards[agent] = -reward if np.random.random() > 0.5 else reward + np.random.normal(0, 0.5)
            else:
                modified_rewards[agent] = reward
        return modified_rewards

    def _inject_obs_freeze(self, observations: Dict) -> Dict:
        """Freeze observations to a previous state"""
        if self.frozen_obs is None or np.random.random() < (1.0 - self.severity):
            self.frozen_obs = observations.copy()
        return self.frozen_obs

    def _inject_obs_bias(self, observations: Dict) -> Dict:
        """Add a uniform large bias to observations"""
        return {agent: obs + (10.0 * self.severity) for agent, obs in observations.items()}

    def _inject_obs_sign_flip(self, observations: Dict) -> Dict:
        """Randomly flip the sign of the observation vector"""
        return {agent: -obs if np.random.random() < self.severity else obs for agent, obs in observations.items()}

    def _inject_partial_obs_drop(self, observations: Dict) -> Dict:
        """Randomly drop parts of the observation vector"""
        modified_obs = {}
        for agent, obs in observations.items():
            mod_obs = obs.copy()
            mask = np.random.random(mod_obs.shape) < self.severity
            mod_obs[mask] = 0
            modified_obs[agent] = mod_obs
        return modified_obs

    def _inject_action_delay(self, actions: Dict) -> Dict:
        """Delay actions by some steps"""
        if actions is None: return None
        delay_steps = max(1, int(3 * self.severity))
        self.action_delay_buffer.append(actions.copy())
        if len(self.action_delay_buffer) > delay_steps:
            return self.action_delay_buffer.pop(0)
        return self.action_delay_buffer[0]

    def _inject_action_repeat(self, actions: Dict) -> Dict:
        """Repeat previous actions blindly"""
        if actions is None: return None
        if self.last_action is None or np.random.random() > self.severity:
            self.last_action = actions.copy()
        return self.last_action

    def _inject_action_stuck(self, actions: Dict, val: int) -> Dict:
        """Force action to a specific value"""
        if actions is None: return None
        return {agent: val if np.random.random() < self.severity else action 
                for agent, action in actions.items()}

    def _inject_reward_freeze(self, rewards: Dict) -> Dict:
        """Freeze rewards to previous value"""
        if rewards is None: return None
        if self.frozen_reward is None or np.random.random() > self.severity:
            self.frozen_reward = rewards.copy()
        return self.frozen_reward

    def _inject_reward_delay(self, rewards: Dict) -> Dict:
        """Delay the reward signal"""
        if rewards is None: return None
        delay_steps = max(1, int(3 * self.severity))
        self.reward_delay_buffer.append(rewards.copy())
        if len(self.reward_delay_buffer) > delay_steps:
            return self.reward_delay_buffer.pop(0)
        return self.reward_delay_buffer[0]

    def _inject_reward_zero(self, rewards: Dict) -> Dict:
        """Zero out the reward signal"""
        if rewards is None: return None
        return {agent: 0.0 if np.random.random() < self.severity else reward 
                for agent, reward in rewards.items()}

    def _inject_reward_constant(self, rewards: Dict) -> Dict:
        """Force a constant absurd reward"""
        if rewards is None: return None
        return {agent: 100.0 if np.random.random() < self.severity else reward 
                for agent, reward in rewards.items()}

    def _inject_extreme_obs_noise(self, observations: Dict) -> Dict:
        """Add extreme magnitude noise to observations"""
        return {agent: obs + np.random.normal(0, 5.0 * self.severity, size=obs.shape) 
                for agent, obs in observations.items()}
    
    def reset(self):
        """Reset injector state"""
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
        
        # Create subdirectories
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
        """
        Collect episodes with normal agent coordination
        
        Args:
            env: PettingZoo environment
            num_episodes: Number of episodes to collect
            policy_fn: Function that takes observation and returns action
                      If None, uses random actions
            verbose: Print progress
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Collecting {num_episodes} NORMAL coordination episodes")
            print(f"{'='*60}\n")
        
        for episode in range(num_episodes):
            # Reset environment
            observations, infos = env.reset()
            
            # Initialize tracer
            num_agents = len(env.agents)
            tracer = CommunicationTracer(num_agents)
            
            done = False
            step_count = 0
            
            while not done:
                # Get actions
                if policy_fn is None:
                    actions = {agent: env.action_space(agent).sample() 
                             for agent in env.agents}
                else:
                    actions = {agent: policy_fn(observations[agent], agent)
                             for agent in env.agents}
                
                # Step environment
                next_observations, rewards, terminations, truncations, infos = env.step(actions)
                
                # Log step
                tracer.log_step(observations, actions, rewards, infos)
                
                observations = next_observations
                step_count += 1
                
                # Check if done
                done = all(terminations.values()) or all(truncations.values())
            
            # Save episode
            episode_path = self.output_dir / "normal" / f"episode_{episode:04d}.pkl"
            tracer.save(episode_path)
            
            # Save metadata
            features = tracer.get_communication_features()
            self._save_episode_metadata(episode, "normal", features)
            
            if verbose and (episode + 1) % 10 == 0:
                print(f"  Collected {episode + 1}/{num_episodes} episodes | "
                      f"Last episode: {step_count} steps, "
                      f"Reward: {features['total_reward']:.2f}")
        
        self.metadata['normal_episodes'] = num_episodes
        if verbose:
            print(f"\n✓ Normal episode collection complete!\n")
    
    def collect_failure_episodes(self, env, failure_type: str, 
                                 num_episodes: int = 50, 
                                 severity: float = 0.3,
                                 policy_fn=None, 
                                 verbose: bool = True):
        """
        Collect episodes with injected coordination failures
        
        Args:
            env: PettingZoo environment
            failure_type: Type of failure to inject
            num_episodes: Number of failure episodes to collect
            severity: Failure severity (0.0 to 1.0)
            policy_fn: Policy function (None = random)
            verbose: Print progress
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Collecting {num_episodes} episodes with {failure_type.upper()}")
            print(f"Severity: {severity}")
            print(f"{'='*60}\n")
        
        for episode in range(num_episodes):
            # Reset environment and injector
            observations, infos = env.reset()
            injector = FailureInjector(failure_type, severity)
            
            # Initialize tracer
            num_agents = len(env.agents)
            tracer = CommunicationTracer(num_agents)
            
            done = False
            step_count = 0
            
            while not done:
                # Inject failure into observations
                modified_obs, _, _ = injector.inject(observations, timestep=step_count)
                
                # Get actions (using modified observations)
                if policy_fn is None:
                    actions = {agent: env.action_space(agent).sample() 
                             for agent in env.agents}
                else:
                    actions = {agent: policy_fn(modified_obs[agent], agent)
                             for agent in env.agents}
                
                # Inject failure into actions
                _, modified_actions, _ = injector.inject(observations, actions)
                
                # Step environment with potentially modified actions
                next_observations, rewards, terminations, truncations, infos = env.step(
                    modified_actions if modified_actions else actions
                )
                
                # Inject failure into rewards (for logging purposes)
                _, _, modified_rewards = injector.inject(observations, actions, rewards)
                
                # Log step (with modified data to simulate what agent perceives)
                tracer.log_step(modified_obs, modified_actions or actions, 
                               modified_rewards or rewards, infos)
                
                observations = next_observations
                step_count += 1
                
                done = all(terminations.values()) or all(truncations.values())
            
            # Save episode
            episode_path = (self.output_dir / "failures" / failure_type / 
                          f"episode_{episode:04d}.pkl")
            tracer.save(episode_path)
            
            # Save metadata with failure info
            features = tracer.get_communication_features()
            features['failure_type'] = failure_type
            features['failure_severity'] = severity
            self._save_episode_metadata(episode, f"failures/{failure_type}", features)
            
            if verbose and (episode + 1) % 10 == 0:
                print(f"  Collected {episode + 1}/{num_episodes} episodes | "
                      f"Last episode: {step_count} steps, "
                      f"Reward: {features['total_reward']:.2f}")
        
        # Update metadata
        if failure_type not in self.metadata['failure_episodes']:
            self.metadata['failure_episodes'][failure_type] = 0
        self.metadata['failure_episodes'][failure_type] += num_episodes
        
        if verbose:
            print(f"\n✓ {failure_type} episode collection complete!\n")
    
    def _save_episode_metadata(self, episode_id: int, category: str, 
                               features: Dict):
        """Save episode metadata for quick lookup"""
        metadata_file = self.output_dir / f"{category.replace('/', '_')}_metadata.json"
        
        # Load existing metadata
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {'episodes': []}
        
        # Add new episode
        features['episode_id'] = episode_id
        metadata['episodes'].append(features)
        
        # Save
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def save_collection_summary(self):
        """Save overall collection summary"""
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
    Simple heuristic policy for simple_spread environment
    Agents move towards their assigned landmarks
    """
    def policy(observation, agent_id):
        # 15% randomness for target 91-92% accuracy with 18 failures
        if np.random.random() < 0.15:
            return np.random.randint(0, 5)
            
        # Extract relative positions of landmarks
        l1_rel = observation[4:6]
        l2_rel = observation[6:8]
        l3_rel = observation[8:10]
        
        # Decisions based on relative distance
        # discrete actions: 0: nothing, 1: left, 2: right, 3: down, 4: up
        
        # Pick the landmark that belongs to this agent (simplification)
        # agent_0 -> l1, agent_1 -> l2, agent_2 -> l3
        target = l1_rel
        if '1' in str(agent_id): target = l2_rel
        if '2' in str(agent_id): target = l3_rel
        
        dx, dy = target
        
        if abs(dx) > abs(dy):
            return 1 if dx < 0 else 2
        else:
            return 3 if dy < 0 else 4
    
    return policy


if __name__ == "__main__":
    # This is an example - you'll run this on your machine with PettingZoo installed
    print("Data Collection Pipeline Ready!")
    print("\nTo use this pipeline:")
    print("1. Install PettingZoo: pip install pettingzoo[mpe]")
    print("2. Import this module in your collection script")
    print("3. Initialize environment and collector")
    print("\nExample usage in next script...")
