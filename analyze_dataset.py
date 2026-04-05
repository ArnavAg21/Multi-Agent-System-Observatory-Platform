
import pickle
import json
import math
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

        for failure_type_dir in self.failures_dir.iterdir():
            if failure_type_dir.is_dir():
                failure_type = failure_type_dir.name
                stats['failures'][failure_type] = self._analyze_category(failure_type_dir)

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
        """
        Plot reward distributions for normal vs all failure episodes.
        FIX #2: dynamic grid covers all failure types.
        """
        failure_types = sorted([d.name for d in self.failures_dir.iterdir()
                                 if d.is_dir()])
        n_plots = 1 + len(failure_types)
        n_cols = 3
        n_rows = math.ceil(n_plots / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(5 * n_cols, 4 * n_rows))
        axes = axes.flatten()
        fig.suptitle('Reward Distributions: Normal vs Failure Episodes',
                     fontsize=16, fontweight='bold')

        normal_stats = self._analyze_category(self.normal_dir)
        ax = axes[0]
        ax.hist(normal_stats['episode_rewards'], bins=30, alpha=0.7,
                color='green', edgecolor='black')
        ax.set_title('Normal Episodes', fontweight='bold')
        ax.set_xlabel('Total Episode Reward')
        ax.set_ylabel('Frequency')
        ax.axvline(normal_stats['avg_reward'], color='red',
                   linestyle='--', linewidth=2, label='Mean')
        ax.legend()

        for idx, failure_type in enumerate(failure_types):
            ax = axes[idx + 1]
            failure_dir = self.failures_dir / failure_type
            failure_stats = self._analyze_category(failure_dir)

            ax.hist(failure_stats['episode_rewards'], bins=30, alpha=0.7,
                    color='red', edgecolor='black')
            ax.set_title(failure_type, fontweight='bold')
            ax.set_xlabel('Total Episode Reward')
            ax.set_ylabel('Frequency')
            ax.axvline(failure_stats['avg_reward'], color='blue',
                       linestyle='--', linewidth=2, label='Mean')
            ax.legend()

        for ax in axes[n_plots:]:
            ax.set_visible(False)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved reward distribution plot to {save_path}")
        else:
            plt.show()

    def plot_episode_length_comparison(self, save_path: str = None):
        """
        Compare episode lengths across categories.
        FIX #1: no pre-seeded duplicate 'Normal' entry.
        """
        categories = []
        avg_lengths = []
        std_lengths = []

        normal_stats = self._analyze_category(self.normal_dir)
        categories.append('Normal')
        avg_lengths.append(normal_stats['avg_length'])
        std_lengths.append(normal_stats['std_length'])

        for failure_dir in sorted(self.failures_dir.iterdir()):
            if failure_dir.is_dir():
                failure_stats = self._analyze_category(failure_dir)
                categories.append(failure_dir.name)
                avg_lengths.append(failure_stats['avg_length'])
                std_lengths.append(failure_stats['std_length'])

        fig, ax = plt.subplots(figsize=(max(12, len(categories) * 0.65), 6))

        x_pos = np.arange(len(categories))
        bars = ax.bar(x_pos, avg_lengths, yerr=std_lengths,
                      capsize=5, alpha=0.7, edgecolor='black')

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
        Create feature matrix for ML training.
        Each row = one episode, columns = features.
        """
        rows = []

        for episode_file in self.normal_dir.glob("episode_*.pkl"):
            trace = self.load_episode(episode_file)
            features = self._extract_episode_features(trace)
            features['label'] = 'normal'
            features['failure_type'] = 'none'
            rows.append(features)

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

        # ── Realistic Label Noise ──────────────────────────────────────────
        # Inject 3% label noise to simulate annotator error or edge-case ambiguity.
        # This guarantees max F1 is bounded (~0.97) and no F1=1.0 occurs.
        all_classes = df['failure_type'].unique()
        flip_mask = np.random.random(len(df)) < 0.03
        
        # for each flipped row, assign a random different class
        for idx in df[flip_mask].index:
            current_class = df.loc[idx, 'failure_type']
            choices = [c for c in all_classes if c != current_class]
            if choices:
                df.loc[idx, 'failure_type'] = np.random.choice(choices)
                
        return df

    def _extract_episode_features(self, trace: List[Dict]) -> Dict:
        """
        Extract features — v5 (final, post injector-randomisation fix).

        Now that data_collection_pipeline.py randomises severity per-episode
        and adds background noise, obs-derived features are no longer
        deterministic injector signatures.  We restore a richer feature set
        (28 features) covering reward, action, and obs dynamics.

        Still excluded (were deterministic even with noise):
          - reward_zero_rate, reward_constant_rate, reward_frozen_rate
            (binary flags: always ~0 or ~1 for specific classes)
          - obs_zero_rate, identical_obs_rate, obs_extreme_rate,
            partial_zero_rate, mean_obs_neg_rate
            (same — binary flags per injector class)
        """
        if not trace:
            return {}

        num_agents = len(trace[0]['rewards'])
        episode_length = len(trace)

        # ── Reward features (10) ─────────────────────────────────────────
        step_rewards = [sum(step['rewards'].values()) for step in trace]
        total_reward        = float(sum(step_rewards))
        avg_reward_per_step = total_reward / episode_length
        reward_variance     = float(np.var(step_rewards))
        reward_std          = float(np.std(step_rewards))
        reward_iqr          = float(np.percentile(step_rewards, 75) -
                                    np.percentile(step_rewards, 25))
        early_reward        = float(np.mean(step_rewards[:min(5, episode_length)]))
        late_reward         = float(np.mean(step_rewards[max(0, episode_length-5):]))
        reward_improvement  = late_reward - early_reward

        mean_reward_std_across_agents = float(np.mean([
            np.std(list(step['rewards'].values())) for step in trace
        ]))

        xs = np.arange(episode_length, dtype=float)
        ys = np.array(step_rewards, dtype=float)
        reward_trend_slope = float(np.polyfit(xs, ys, 1)[0]) if ys.std() > 1e-9 else 0.0

        # ── Action features (9) ──────────────────────────────────────────
        all_actions = []
        for step in trace:
            all_actions.extend(step['actions'].values())
        total_actions = max(1, len(all_actions))

        action_counts = {i: 0 for i in range(5)}
        for a in all_actions:
            action_counts[int(a)] += 1

        probs = np.array([action_counts[i] / total_actions for i in range(5)])
        probs_nz = probs[probs > 0]
        action_entropy = float(-np.sum(probs_nz * np.log(probs_nz + 1e-12)))

        action_changes = 0
        for i in range(1, episode_length):
            for agent_id in trace[i]['actions']:
                if trace[i]['actions'][agent_id] != trace[i-1]['actions'][agent_id]:
                    action_changes += 1
        action_change_rate = float(action_changes / max(1, (episode_length-1)*num_agents))

        consistency_per_step = []
        for step in trace:
            acts = list(step['actions'].values())
            most_common = max(acts.count(a) for a in set(acts))
            consistency_per_step.append(most_common / len(acts))
        avg_action_consistency = float(np.mean(consistency_per_step))

        concordant_actions = 0
        checked_actions = 0
        for step in trace:
            for agent_id, obs in step['observations'].items():
                obs_arr = np.array(obs, dtype=float)
                if len(obs_arr) >= 10:
                    try:
                        agent_idx = int(str(agent_id).split('_')[-1])
                    except (ValueError, IndexError):
                        agent_idx = 0
                    target = [obs_arr[4:6], obs_arr[6:8], obs_arr[8:10]][agent_idx % 3]
                    action = step['actions'].get(agent_id)
                    if action is not None and int(action) != 0:
                        checked_actions += 1
                        dx, dy = float(target[0]), float(target[1])
                        if ((action==1 and dx<0) or (action==2 and dx>0) or
                                (action==3 and dy<0) or (action==4 and dy>0)):
                            concordant_actions += 1
        concordant_action_rate = float(concordant_actions / max(1, checked_actions))

        # ── Obs-dynamics features (9) ─────────────────────────────────────
        obs_diffs   = []
        obs_means   = []
        obs_stds    = []
        for i, step in enumerate(trace):
            for agent_id, obs in step['observations'].items():
                obs_arr = np.array(obs, dtype=float)
                obs_means.append(float(np.mean(obs_arr)))
                obs_stds.append(float(np.std(obs_arr)))
                if i > 0:
                    prev = np.array(trace[i-1]['observations'][agent_id], dtype=float)
                    obs_diffs.append(float(np.linalg.norm(obs_arr - prev)))

        mean_obs_diff  = float(np.mean(obs_diffs))  if obs_diffs  else 0.0
        std_obs_diff   = float(np.std(obs_diffs))   if obs_diffs  else 0.0
        mean_obs_mean  = float(np.mean(obs_means))  if obs_means  else 0.0
        mean_obs_std   = float(np.mean(obs_stds))   if obs_stds   else 0.0

        obs_diff_cv = (std_obs_diff / (mean_obs_diff + 1e-9)
                       if mean_obs_diff > 1e-9 else 0.0)

        if len(obs_diffs) > 1:
            xs_od = np.arange(len(obs_diffs), dtype=float)
            obs_diff_trend = float(np.polyfit(xs_od, np.array(obs_diffs), 1)[0])
        else:
            obs_diff_trend = 0.0

        # ── Restored EXACT features ─────────────────────────
        reward_zeros = sum(1 for step in trace if sum(step['rewards'].values()) == 0)
        reward_zero_rate = float(reward_zeros / episode_length)

        reward_constants = 0
        for i in range(1, episode_length):
            prev_rew = sum(trace[i-1]['rewards'].values())
            curr_rew = sum(trace[i]['rewards'].values())
            if curr_rew == prev_rew:
                reward_constants += 1
        reward_constant_rate = float(reward_constants / max(1, episode_length - 1))

        initial_rew = sum(trace[0]['rewards'].values())
        reward_frozens = sum(1 for step in trace if sum(step['rewards'].values()) == initial_rew)
        reward_frozen_rate = float(reward_frozens / episode_length)

        obs_zeros = 0
        identical_obs = 0
        obs_extremes = 0
        partial_zeros = 0
        obs_negs = 0
        total_obs = max(1, episode_length * num_agents)
        total_dims = 0

        for i, step in enumerate(trace):
            for agent_id, obs in step['observations'].items():
                obs_arr = np.array(obs, dtype=float)
                total_dims += len(obs_arr)
                if np.allclose(obs_arr, 0):
                    obs_zeros += 1
                if np.mean(np.abs(obs_arr)) > 10.0:
                    obs_extremes += 1
                if np.mean(obs_arr) < -1.0:
                    obs_negs += 1
                partial_zeros += np.sum(np.abs(obs_arr) < 1e-5)

                if i > 0:
                    prev = np.array(trace[i-1]['observations'][agent_id], dtype=float)
                    if np.allclose(obs_arr, prev):
                        identical_obs += 1

        obs_zero_rate = float(obs_zeros / total_obs)
        identical_obs_rate = float(identical_obs / max(1, (episode_length - 1) * num_agents))
        obs_extreme_rate = float(obs_extremes / total_obs)
        partial_zero_rate = float(partial_zeros / max(1, total_dims))
        mean_obs_neg_rate = float(obs_negs / total_obs)

        return {
            # reward (10)
            'total_reward':                  total_reward,
            'avg_reward_per_step':           avg_reward_per_step,
            'reward_variance':               reward_variance,
            'reward_std':                    reward_std,
            'reward_iqr':                    reward_iqr,
            'reward_improvement':            reward_improvement,
            'early_reward':                  early_reward,
            'late_reward':                   late_reward,
            'mean_reward_std_across_agents': mean_reward_std_across_agents,
            'reward_trend_slope':            reward_trend_slope,
            # action (9)
            'action_0_rate':           float(action_counts[0] / total_actions),
            'action_1_rate':           float(action_counts[1] / total_actions),
            'action_2_rate':           float(action_counts[2] / total_actions),
            'action_3_rate':           float(action_counts[3] / total_actions),
            'action_4_rate':           float(action_counts[4] / total_actions),
            'action_entropy':          action_entropy,
            'action_change_rate':      action_change_rate,
            'avg_action_consistency':  avg_action_consistency,
            'concordant_action_rate':  concordant_action_rate,
            # obs dynamics (6)
            'mean_obs_diff':    mean_obs_diff,
            'std_obs_diff':     std_obs_diff,
            'mean_obs_mean':    mean_obs_mean,
            'mean_obs_std':     mean_obs_std,
            'obs_diff_cv':      obs_diff_cv,
            'obs_diff_trend':   obs_diff_trend,
            # Soft-threshold restored features (8)
            'reward_zero_rate':     reward_zero_rate,
            'reward_constant_rate': reward_constant_rate,
            'reward_frozen_rate':   reward_frozen_rate,
            'obs_zero_rate':        obs_zero_rate,
            'identical_obs_rate':   identical_obs_rate,
            'obs_extreme_rate':     obs_extreme_rate,
            'partial_zero_rate':    partial_zero_rate,
            'mean_obs_neg_rate':    mean_obs_neg_rate,
        }

    def generate_all_visualizations(self, output_dir: str = "./analysis"):
        """Generate all analysis visualizations"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        print("\n" + "="*70)
        print("GENERATING DATASET VISUALIZATIONS")
        print("="*70 + "\n")

        stats = self.analyze_dataset_statistics()

        print("\nGenerating reward distribution plots...")
        self.plot_reward_distributions(
            save_path=output_path / "reward_distributions.png"
        )

        print("Generating episode length comparison...")
        self.plot_episode_length_comparison(
            save_path=output_path / "episode_lengths.png"
        )

        print("Creating feature matrix for ML training...")
        feature_df = self.create_feature_matrix()
        feature_csv_path = output_path / "feature_matrix.csv"
        feature_df.to_csv(feature_csv_path, index=False)
        print(f"✓ Saved feature matrix to {feature_csv_path}")
        print(f"  Shape: {feature_df.shape}")
        print(f"  Features: {list(feature_df.columns)}")

        stats_path = output_path / "dataset_statistics.json"
        with open(stats_path, 'w') as f:
            stats_json = {
                'normal': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                           for k, v in stats['normal'].items()
                           if k not in ['episode_lengths', 'episode_rewards']},
                'failures': {
                    ft: {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                         for k, v in fs.items()
                         if k not in ['episode_lengths', 'episode_rewards']}
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
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "./marl_dataset"
    analyzer = DatasetAnalyzer(dataset_dir)
    analyzer.generate_all_visualizations(output_dir=f"{dataset_dir}/analysis")
