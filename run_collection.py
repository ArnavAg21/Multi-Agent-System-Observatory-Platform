from pettingzoo.mpe import simple_spread_v3
from data_collection_pipeline import DataCollector, FailureInjector

def main():
    print("Initializing environment for data collection...")
    env = simple_spread_v3.parallel_env(N=3, max_cycles=25, continuous_actions=False)
    collector = DataCollector(output_dir="./marl_dataset")
    
    # Collect normal episodes
    collector.collect_normal_episodes(env, num_episodes=300, verbose=True)
    
    # Collect some failure episodes
    failures = FailureInjector.FAILURE_TYPES
    for failure in failures:
        collector.collect_failure_episodes(env, failure_type=failure, num_episodes=300, severity=0.3, verbose=True)
        
    collector.save_collection_summary()
    env.close()
    print("Data collection completed successfully.")

if __name__ == "__main__":
    main()
