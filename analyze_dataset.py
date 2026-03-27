import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
import pandas as pd


class DatasetAnalyzer:
    """Analyze collected MARL coordination dataset"""
    
    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)
        self.normal_dir = self.dataset_dir / "normal"
        self.failures_dir = self.dataset_dir / "failures"
        
    def load_episode(self, filepath: str) -> List[Dict]:
        """Load single episode trace"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def analyze_dataset_statistics(self):
        """Generate overall dataset statistics"""
        print("\n" + "="*70)
        print("DATASET STATISTICS")
        print("="*70 + "\n")
        
        stats = {
            'normal': self._analyze_category(self.normal_dir),
            'failures': {}
        }
        
        # Analyze each failure type
        for failure_type_dir in self.failures_dir.iterdir():
            if failure_type_dir.is_dir():
                failure_type = failure_type_dir.name
                stats['failures'][failure_type] = self._analyze_category(failure_type_dir)
        
        # Print statistics
        print(f"Normal Episodes: {stats['normal']['num_episodes']}")
        print(f"  Avg episode length: {stats['normal']['avg_length']:.2f} steps")
        print(f"  Avg total reward: {stats['normal']['avg_reward']:.2f}")
        print(f"  Reward std: {stats['normal']['std_reward']:.2f}")
        
        print("\nFailure Episodes:")
        for failure_type, failure_stats in stats['failures'].items():
            print(f"\n  {failure_type}:")
            print(f"    Episodes: {failure_stats['num_episodes']}")
            print(f"    Avg length: {failure_stats['avg_length']:.2f} steps")
            print(f"    Avg reward: {failure_stats['avg_reward']:.2f}")
            print(f"    Reward std: {failure_stats['std_reward']:.2f}")
        
        return stats
    
    def _analyze_category(self, category_dir: Path) -> Dict:
        """Analyze episodes in a category"""
        episode_files = list(category_dir.glob("episode_*.pkl"))
        
        lengths = []
        rewards = []
        
        for episode_file in episode_files:
            trace = self.load_episode(episode_file)
            lengths.append(len(trace))
            
            # Calculate total reward
            total_reward = sum(sum(step['rewards'].values()) for step in trace)
            rewards.append(total_reward)
        
        return {
            'num_episodes': len(episode_files),
            'avg_length': np.mean(lengths) if lengths else 0,
            'std_length': np.std(lengths) if lengths else 0,
            'avg_reward': np.mean(rewards) if rewards else 0,
            'std_reward': np.std(rewards) if rewards else 0,
            'episode_lengths': lengths,
            'episode_rewards': rewards
        }
    
    def plot_reward_distributions(self, save_path: str = None):
        """Plot reward distributions for normal vs failure episodes"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Reward Distributions: Normal vs Failure Episodes', 
                     fontsize=16, fontweight='bold')
        
        # Load normal episodes
        normal_stats = self._analyze_category(self.normal_dir)
        
        # Plot normal distribution
        ax = axes[0, 0]
        ax.hist(normal_stats['episode_rewards'], bins=30, alpha=0.7, 
                color='green', edgecolor='black')
        ax.set_title('Normal Episodes', fontweight='bold')
        ax.set_xlabel('Total Episode Reward')
        ax.set_ylabel('Frequency')
        ax.axvline(normal_stats['avg_reward'], color='red', 
                   linestyle='--', linewidth=2, label='Mean')
        ax.legend()
        
        # Plot each failure type
        failure_types = [d.name for d in self.failures_dir.iterdir() if d.is_dir()]
        
        for idx, failure_type in enumerate(failure_types[:5]):  # Max 5 failure types
            row = (idx + 1) // 3
            col = (idx + 1) % 3
            ax = axes[row, col]
            
            failure_dir = self.failures_dir / failure_type
            failure_stats = self._analyze_category(failure_dir)
            
            ax.hist(failure_stats['episode_rewards'], bins=30, alpha=0.7,
                   color='red', edgecolor='black')
            ax.set_title(f'{failure_type}', fontweight='bold')
            ax.set_xlabel('Total Episode Reward')
            ax.set_ylabel('Frequency')
            ax.axvline(failure_stats['avg_reward'], color='blue',
                      linestyle='--', linewidth=2, label='Mean')
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved reward distribution plot to {save_path}")
        else:
            plt.show()
    
    def plot_episode_length_comparison(self, save_path: str = None):
        """Compare episode lengths across categories"""
        categories = ['Normal']
        avg_lengths = []
        std_lengths = []
        
        # Normal episodes
        normal_stats = self._analyze_category(self.normal_dir)
        categories.append('Normal')
        avg_lengths.append(normal_stats['avg_length'])
        std_lengths.append(normal_stats['std_length'])
        
        # Failure episodes
        for failure_dir in self.failures_dir.iterdir():
            if failure_dir.is_dir():
                failure_stats = self._analyze_category(failure_dir)
                categories.append(failure_dir.name)
                avg_lengths.append(failure_stats['avg_length'])
                std_lengths.append(failure_stats['std_length'])
        
        # Remove duplicate 'Normal' if it exists
        if categories[0] == 'Normal' and len(categories) > 1:
            categories = categories[1:]
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x_pos = np.arange(len(categories))
        bars = ax.bar(x_pos, avg_lengths, yerr=std_lengths, 
                     capsize=5, alpha=0.7, edgecolor='black')
        
        # Color normal differently
        bars[0].set_color('green')
        for bar in bars[1:]:
            bar.set_color('red')
        
        ax.set_xlabel('Episode Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Average Episode Length (steps)', fontsize=12, fontweight='bold')
        ax.set_title('Episode Length Comparison: Normal vs Failures', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved episode length plot to {save_path}")
        else:
            plt.show()
    
    def create_feature_matrix(self) -> pd.DataFrame:
        """
        Create feature matrix for ML training
        Each row = one episode, columns = features
        """
        rows = []
        
        # Process normal episodes
        for episode_file in self.normal_dir.glob("episode_*.pkl"):
            trace = self.load_episode(episode_file)
            features = self._extract_episode_features(trace)
            features['label'] = 'normal'
            features['failure_type'] = 'none'
            rows.append(features)
        
        # Process failure episodes
        for failure_dir in self.failures_dir.iterdir():
            if failure_dir.is_dir():
                failure_type = failure_dir.name
                for episode_file in failure_dir.glob("episode_*.pkl"):
                    trace = self.load_episode(episode_file)
                    features = self._extract_episode_features(trace)
                    features['label'] = 'failure'
                    features['failure_type'] = failure_type
                    rows.append(features)
        
        df = pd.DataFrame(rows)
        return df
    
    def _extract_episode_features(self, trace: List[Dict]) -> Dict:
        """Extract features from episode trace for ML"""
        if not trace:
            return {}
        
        # Basic statistics
        episode_length = len(trace)
        total_reward = sum(sum(step['rewards'].values()) for step in trace)
        avg_reward_per_step = total_reward / episode_length
        
        # Reward variance
        step_rewards = [sum(step['rewards'].values()) for step in trace]
        reward_variance = np.var(step_rewards)
        
        # Action diversity
        all_actions = []
        for step in trace:
            all_actions.extend(step['actions'].values())
        action_diversity = len(set(all_actions)) / len(all_actions) if all_actions else 0
        
        # Action consistency (how often agents take same action)
        action_consistency = []
        for step in trace:
            actions = list(step['actions'].values())
            most_common_count = max([actions.count(a) for a in set(actions)])
            action_consistency.append(most_common_count / len(actions))
        avg_action_consistency = np.mean(action_consistency)
        
        # Reward progression (early vs late)
        early_reward = np.mean([sum(trace[i]['rewards'].values()) 
                               for i in range(min(5, len(trace)))])
        late_reward = np.mean([sum(trace[i]['rewards'].values()) 
                              for i in range(max(0, len(trace)-5), len(trace))])
        reward_improvement = late_reward - early_reward
        
        # --- NEW FEATURES to better distinguish failure types ---
        obs_zero_count = 0
        total_obs_count = 0
        obs_diffs = []
        identical_obs_count = 0
        action_changes = 0
        reward_std_across_agents = []
        
        # Lagged correlation features
        lagged_diff_2 = []
        
        # Policy concordance - does agent move towards its "natural" landmark?
        concordant_actions = 0
        
        # Start-of-episode lag (specifically for communication_delay)
        start_identical = 0
        
        # ++++++ EXTRA FEATURES FOR 18 CLASSES ++++++
        obs_mean_val = 0.0
        obs_neg_count = 0
        total_obs_elements = 0
        partial_zero_count = 0
        action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        total_actions = 0
        reward_zero_count = 0
        reward_constant_count = 0
        reward_frozen_count = 0
        obs_extreme_count = 0
        
        for i in range(len(trace)):
            step = trace[i]
            
            # Reward std across agents - detects reward_misalignment
            agent_rewards = list(step['rewards'].values())
            reward_std_across_agents.append(np.std(agent_rewards))
            
            # Extended reward features
            for r in agent_rewards:
                if abs(r) < 1e-6:
                    reward_zero_count += 1
                if abs(r - 100.0) < 1e-6:
                    reward_constant_count += 1
            if i > 0:
                prev_agent_rewards = list(trace[i-1]['rewards'].values())
                if np.allclose(agent_rewards, prev_agent_rewards):
                    reward_frozen_count += 1

            # Extended action features
            for agent_id, act in step['actions'].items():
                act_int = int(act)
                if act_int in action_counts:
                    action_counts[act_int] += 1
                total_actions += 1
            
            for agent_id, obs in step['observations'].items():
                obs_arr = np.array(obs)
                total_obs_count += 1
                
                # Extended obs statistics
                obs_mean_val += np.sum(obs_arr)
                total_obs_elements += obs_arr.size
                obs_neg_count += np.sum(obs_arr < -1e-6)
                partial_zero_count += np.sum(np.abs(obs_arr) < 1e-6)
                if np.max(np.abs(obs_arr)) > 4.0:
                    obs_extreme_count += 1
                
                # Zero observations - detects message_drop
                if not np.any(obs_arr):  
                    obs_zero_count += 1
                    
                if i > 0:
                    prev_obs = np.array(trace[i-1]['observations'][agent_id])
                    diff = np.linalg.norm(obs_arr - prev_obs)
                    obs_diffs.append(diff)
                    
                    # Identical non-zero observations - detects delay
                    if diff < 1e-6 and np.any(obs_arr):
                        identical_obs_count += 1
                        if i < 5: # Specifically check start of episode
                            start_identical += 1
                
                # Multi-step lag
                if i > 1:
                    prev_2 = np.array(trace[i-2]['observations'][agent_id])
                    lagged_diff_2.append(np.linalg.norm(obs_arr - prev_2))

                # Check action concordance (simple_spread landmark logic)
                # landmark_rel_positions start at index 4
                if len(obs_arr) >= 10:
                    l1_rel = obs_arr[4:6]
                    l2_rel = obs_arr[6:8]
                    l3_rel = obs_arr[8:10]
                    target = l1_rel
                    if '1' in str(agent_id): target = l2_rel
                    if '2' in str(agent_id): target = l3_rel
                    
                    action = step['actions'].get(agent_id)
                    if action is not None and action != 0:
                        dx, dy = target
                        # 1: left, 2: right, 3: down, 4: up
                        is_concordant = False
                        if action == 1 and dx < 0: is_concordant = True
                        if action == 2 and dx > 0: is_concordant = True
                        if action == 3 and dy < 0: is_concordant = True
                        if action == 4 and dy > 0: is_concordant = True
                        if is_concordant:
                            concordant_actions += 1
            
            if i > 0:
                for agent_id, action in step['actions'].items():
                    if action != trace[i-1]['actions'][agent_id]:
                        action_changes += 1

        obs_zero_rate = obs_zero_count / total_obs_count if total_obs_count > 0 else 0
        mean_obs_diff = np.mean(obs_diffs) if obs_diffs else 0
        
        non_zero_obs_count = total_obs_count - obs_zero_count
        identical_obs_rate = identical_obs_count / non_zero_obs_count if non_zero_obs_count > 0 else 0
        
        total_action_count_over_time = len(trace) * len(step['actions']) - len(step['actions'])
        action_change_rate = action_changes / total_action_count_over_time if total_action_count_over_time > 0 else 0 
        mean_reward_std = np.mean(reward_std_across_agents) if reward_std_across_agents else 0
        
        lag_ratio = np.mean(lagged_diff_2) / (mean_obs_diff + 1e-6) if lagged_diff_2 else 0
        # ------------------------------------------------------
        
        return {
            'episode_length': episode_length,
            'total_reward': total_reward,
            'avg_reward_per_step': avg_reward_per_step,
            'reward_variance': reward_variance,
            'action_diversity': action_diversity,
            'action_consistency': avg_action_consistency,
            'reward_improvement': reward_improvement,
            'early_reward': early_reward,
            'late_reward': late_reward,
            'obs_zero_rate': obs_zero_rate,
            'mean_obs_diff': mean_obs_diff,
            'identical_obs_rate': identical_obs_rate,
            'action_change_rate': action_change_rate,
            'mean_reward_std': mean_reward_std,
            'lag_ratio': lag_ratio,
            'obs_mean': float(obs_mean_val / max(1, total_obs_elements)),
            'obs_neg_rate': float(obs_neg_count / max(1, total_obs_elements)),
            'partial_zero_rate': float(partial_zero_count / max(1, total_obs_elements)),
            'action_0_rate': float(action_counts.get(0, 0) / max(1, total_actions)),
            'action_1_rate': float(action_counts.get(1, 0) / max(1, total_actions)),
            'action_2_rate': float(action_counts.get(2, 0) / max(1, total_actions)),
            'action_3_rate': float(action_counts.get(3, 0) / max(1, total_actions)),
            'action_4_rate': float(action_counts.get(4, 0) / max(1, total_actions)),
            'concordant_action_rate': float(concordant_actions / max(1, total_actions)),
            'reward_zero_rate': float(reward_zero_count / max(1, len(trace) * max(1, len(step['rewards'])))),
            'reward_constant_rate': float(reward_constant_count / max(1, len(trace) * max(1, len(step['rewards'])))),
            'reward_frozen_rate': float(reward_frozen_count / max(1, len(trace) - 1)),
            'obs_extreme_rate': float(obs_extreme_count / max(1, total_obs_count))
        }
    
    def generate_all_visualizations(self, output_dir: str = "./analysis"):
        """Generate all analysis visualizations"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        print("\n" + "="*70)
        print("GENERATING DATASET VISUALIZATIONS")
        print("="*70 + "\n")
        
        # 1. Dataset statistics
        stats = self.analyze_dataset_statistics()
        
        # 2. Reward distributions
        print("\nGenerating reward distribution plots...")
        self.plot_reward_distributions(
            save_path=output_path / "reward_distributions.png"
        )
        
        # 3. Episode length comparison
        print("Generating episode length comparison...")
        self.plot_episode_length_comparison(
            save_path=output_path / "episode_lengths.png"
        )
        
        # 4. Create feature matrix for ML
        print("Creating feature matrix for ML training...")
        feature_df = self.create_feature_matrix()
        feature_csv_path = output_path / "feature_matrix.csv"
        feature_df.to_csv(feature_csv_path, index=False)
        print(f"✓ Saved feature matrix to {feature_csv_path}")
        print(f"  Shape: {feature_df.shape}")
        print(f"  Features: {list(feature_df.columns)}")
        
        # 5. Save statistics as JSON
        stats_path = output_path / "dataset_statistics.json"
        with open(stats_path, 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            stats_json = {
                'normal': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                          for k, v in stats['normal'].items() if k not in ['episode_lengths', 'episode_rewards']},
                'failures': {
                    ft: {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                        for k, v in fs.items() if k not in ['episode_lengths', 'episode_rewards']}
                    for ft, fs in stats['failures'].items()
                }
            }
            json.dump(stats_json, f, indent=2)
        print(f"✓ Saved statistics to {stats_path}")
        
        print("\n" + "="*70)
        print("ANALYSIS COMPLETE!")
        print("="*70)
        print(f"\nAll outputs saved to: {output_path}")


if __name__ == "__main__":
    import sys
    
    # Usage: python analyze_dataset.py [dataset_dir]
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "./marl_dataset"
    
    analyzer = DatasetAnalyzer(dataset_dir)
    analyzer.generate_all_visualizations(output_dir=f"{dataset_dir}/analysis")
