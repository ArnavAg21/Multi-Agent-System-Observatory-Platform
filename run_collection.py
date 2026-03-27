from pettingzoo.mpe import simple_spread_v3
from data_collection_pipeline import DataCollector, FailureInjector, create_simple_policy


def main():
    """Main data collection workflow"""
    
    print("\n" + "="*70)
    print("MARL OBSERVATORY PLATFORM - DATA COLLECTION")
    print("="*70 + "\n")
    
    # Configuration
    NUM_NORMAL_EPISODES = 800       
    NUM_FAILURE_EPISODES = 150      
    FAILURE_SEVERITY = 0.34         # 1-step delay
    OUTPUT_DIR = "./marl_dataset"
    
    # Initialize environment
    print("Initializing PettingZoo Simple Spread environment...")
    env = simple_spread_v3.parallel_env(
        N=3,                    # 3 agents
        local_ratio=0.5,
        max_cycles=25,          # 25 timesteps per episode
        continuous_actions=False
    )
    
    # Initialize data collector
    collector = DataCollector(output_dir=OUTPUT_DIR)
    
    # Optional: Create a simple policy (or use None for random)
    policy = create_simple_policy()  # or set to None for random actions
    
    # ========================================================================
    # PHASE 1: Collect Normal Coordination Episodes
    # ========================================================================
    print("\n" + "="*70)
    print("PHASE 1: COLLECTING NORMAL COORDINATION DATA")
    print("="*70)
    
    collector.collect_normal_episodes(
        env=env,
        num_episodes=NUM_NORMAL_EPISODES,
        policy_fn=policy,  # Use heuristic policy
        verbose=True
    )
    
    # ========================================================================
    # PHASE 2: Collect Failure Episodes for Each Failure Type
    # ========================================================================
    print("\n" + "="*70)
    print("PHASE 2: COLLECTING COORDINATION FAILURE DATA")
    print("="*70)
    
    failure_types = FailureInjector.FAILURE_TYPES
    
    for failure_type in failure_types:
        collector.collect_failure_episodes(
            env=env,
            failure_type=failure_type,
            num_episodes=NUM_FAILURE_EPISODES,
            severity=FAILURE_SEVERITY,
            policy_fn=policy, # Use same policy
            verbose=True
        )
    
    # ========================================================================
    # PHASE 3: Save Summary
    # ========================================================================
    collector.save_collection_summary()
    
    env.close()
    
    print("\n✓ DATA COLLECTION COMPLETE!")
    print(f"\nDataset saved to: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("1. Run analyze_dataset.py to visualize the collected data")
    print("2. Run train_detector.py to train failure detection model")
    print("3. Run evaluate_observatory.py to test the platform")


if __name__ == "__main__":
    main()
