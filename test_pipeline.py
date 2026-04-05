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

        print("Initializing environment...")
        env = simple_spread_v3.parallel_env(N=3, max_cycles=25, continuous_actions=False)
        print("✓ Environment initialized")

        print("\nInitializing data collector...")
        test_dir = "./test_dataset"
        collector = DataCollector(output_dir=test_dir)
        print(f"✓ Collector initialized (output: {test_dir})")

        n_normal = 2
        n_failure = 1  # 1 episode per failure type to keep tests fast

        print(f"\nCollecting {n_normal} normal episodes...")
        collector.collect_normal_episodes(env, num_episodes=n_normal, verbose=False)
        print("✓ Normal episodes collected")

        from data_collection_pipeline import FailureInjector
        failure_types = FailureInjector.FAILURE_TYPES

        print(f"\nCollecting {n_failure} failure episode(s) for each of the {len(failure_types)} types...")
        for failure_type in failure_types:
            collector.collect_failure_episodes(
                env,
                failure_type=failure_type,
                num_episodes=n_failure,
                severity=0.3,
                verbose=False
            )
        print("✓ All failure episodes collected")

        collector.save_collection_summary()

        print("\nVerifying output files...")
        normal_dir = Path(test_dir) / "normal"

        normal_files = list(normal_dir.glob("episode_*.pkl"))
        assert len(normal_files) == n_normal, (
            f"Expected {n_normal} normal files, found {len(normal_files)}")

        total_failure_files = 0
        for failure_type in failure_types:
            failure_dir = Path(test_dir) / "failures" / failure_type
            failure_files = list(failure_dir.glob("episode_*.pkl"))
            assert len(failure_files) == n_failure, (
                f"Expected {n_failure} failure files for {failure_type}, found {len(failure_files)}")
            total_failure_files += len(failure_files)

        print(f"✓ Found {len(normal_files)} normal episodes")
        print(f"✓ Found {total_failure_files} failure episodes across all {len(failure_types)} types")

        env.close()

        # Store counts for downstream tests
        test_pipeline._normal_count = n_normal
        test_pipeline._failure_count_total = total_failure_files

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

        # FIX #1: derive expected count from what test_pipeline actually collected
        # instead of a hardcoded literal (4).
        n_normal = getattr(test_pipeline, '_normal_count', 2)
        n_failure_total = getattr(test_pipeline, '_failure_count_total', 18 * 1)
        expected_rows = n_normal + n_failure_total

        assert len(feature_df) == expected_rows, (
            f"Expected {expected_rows} episodes, got {len(feature_df)}")
        print(f"✓ Feature matrix created "
              f"({feature_df.shape[0]} episodes, {feature_df.shape[1]} features)")

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

    if not test_imports():
        print("\n⚠ Please install missing packages before continuing")
        sys.exit(1)

    pipeline_ok = test_pipeline()

    analysis_ok = False
    if pipeline_ok:
        analysis_ok = test_analysis()

    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"✓ Imports: PASSED")
    print(f"{'✓' if pipeline_ok else '✗'} Data Collection: "
          f"{'PASSED' if pipeline_ok else 'FAILED'}")
    print(f"{'✓' if analysis_ok else '✗'} Analysis: "
          f"{'PASSED' if analysis_ok else 'FAILED'}")

    if pipeline_ok:
        print("\n" + "="*70)
        cleanup()

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
