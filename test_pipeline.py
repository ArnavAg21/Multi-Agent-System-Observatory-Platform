import sys
from pathlib import Path


def test_imports():
    """Test that all required packages are installed"""
    print("\n" + "="*70)
    print("TESTING IMPORTS")
    print("="*70 + "\n")
    
    required = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
    ]
    
    missing = []
    for module, name in required:
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - NOT INSTALLED")
            missing.append(name)
    
    # Test PettingZoo separately (it's critical)
    try:
        from pettingzoo.mpe import simple_spread_v3
        print(f"✓ PettingZoo")
    except ImportError:
        print(f"✗ PettingZoo - NOT INSTALLED")
        missing.append("PettingZoo")
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("\nPlease install with:")
        print("  pip install -r requirements.txt")
        return False
    
    print("\n✓ All packages installed successfully!")
    return True


def test_pipeline():
    """Test data collection pipeline with minimal run"""
    print("\n" + "="*70)
    print("TESTING DATA COLLECTION PIPELINE")
    print("="*70 + "\n")
    
    try:
        from pettingzoo.mpe import simple_spread_v3
        from data_collection_pipeline import DataCollector
        
        # Initialize environment
        print("Initializing environment...")
        env = simple_spread_v3.parallel_env(N=3, max_cycles=25, continuous_actions=False)
        print("✓ Environment initialized")
        
        # Initialize collector
        print("\nInitializing data collector...")
        test_dir = "./test_dataset"
        collector = DataCollector(output_dir=test_dir)
        print(f"✓ Collector initialized (output: {test_dir})")
        
        # Collect 2 normal episodes
        print("\nCollecting 2 normal episodes...")
        collector.collect_normal_episodes(env, num_episodes=2, verbose=False)
        print("✓ Normal episodes collected")
        
        # Collect 2 failure episodes
        print("\nCollecting 2 failure episodes (communication_delay)...")
        collector.collect_failure_episodes(
            env, 
            failure_type='communication_delay',
            num_episodes=2,
            severity=0.3,
            verbose=False
        )
        print("✓ Failure episodes collected")
        
        # Save summary
        collector.save_collection_summary()
        
        # Verify files exist
        print("\nVerifying output files...")
        normal_dir = Path(test_dir) / "normal"
        failure_dir = Path(test_dir) / "failures" / "communication_delay"
        
        normal_files = list(normal_dir.glob("episode_*.pkl"))
        failure_files = list(failure_dir.glob("episode_*.pkl"))
        
        assert len(normal_files) == 2, f"Expected 2 normal files, found {len(normal_files)}"
        assert len(failure_files) == 2, f"Expected 2 failure files, found {len(failure_files)}"
        
        print(f"✓ Found {len(normal_files)} normal episodes")
        print(f"✓ Found {len(failure_files)} failure episodes")
        
        env.close()
        
        print("\n✓ Pipeline test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Pipeline test FAILED!")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_analysis():
    """Test dataset analysis"""
    print("\n" + "="*70)
    print("TESTING ANALYSIS PIPELINE")
    print("="*70 + "\n")
    
    try:
        from analyze_dataset import DatasetAnalyzer
        import pandas as pd
        
        test_dir = "./test_dataset"
        
        print("Initializing analyzer...")
        analyzer = DatasetAnalyzer(test_dir)
        print("✓ Analyzer initialized")
        
        print("\nGenerating statistics...")
        stats = analyzer.analyze_dataset_statistics()
        print("✓ Statistics generated")
        
        print("\nCreating feature matrix...")
        feature_df = analyzer.create_feature_matrix()
        assert isinstance(feature_df, pd.DataFrame), "Feature matrix should be DataFrame"
        assert len(feature_df) == 4, f"Expected 4 episodes, got {len(feature_df)}"
        print(f"✓ Feature matrix created ({feature_df.shape[0]} episodes, {feature_df.shape[1]} features)")
        
        print("\n✓ Analysis test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Analysis test FAILED!")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def cleanup():
    """Clean up test files"""
    import shutil
    test_dir = Path("./test_dataset")
    if test_dir.exists():
        shutil.rmtree(test_dir)
        print(f"\n✓ Cleaned up test directory: {test_dir}")


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("MARL OBSERVATORY PLATFORM - PIPELINE VERIFICATION")
    print("="*70)
    
    # Test 1: Imports
    if not test_imports():
        print("\n⚠ Please install missing packages before continuing")
        sys.exit(1)
    
    # Test 2: Pipeline
    pipeline_ok = test_pipeline()
    
    # Test 3: Analysis (only if pipeline succeeded)
    analysis_ok = False
    if pipeline_ok:
        analysis_ok = test_analysis()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"{'✓' if True else '✗'} Imports: PASSED")
    print(f"{'✓' if pipeline_ok else '✗'} Data Collection: {'PASSED' if pipeline_ok else 'FAILED'}")
    print(f"{'✓' if analysis_ok else '✗'} Analysis: {'PASSED' if analysis_ok else 'FAILED'}")
    
    # Cleanup
    if pipeline_ok:
        print("\n" + "="*70)
        cleanup()
    
    # Final message
    print("\n" + "="*70)
    if pipeline_ok and analysis_ok:
        print("🎉 ALL TESTS PASSED!")
        print("\nYour pipeline is ready to use!")
        print("\nNext step: Run full data collection")
        print("  python run_collection.py")
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease check error messages above and fix issues")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
