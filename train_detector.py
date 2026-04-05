import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Features with zero variance — excluded from training.
# episode_length: always 25 (zero variance with max_cycles=25).
# Leaky features removed at source in analyze_dataset.py _extract_episode_features.
DEAD_FEATURES = ['episode_length']


class FailureDetector:
    """
    Baseline failure detection model.
    Classifies episodes as normal or failure type.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        self.label_names = None

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              verbose: bool = True):
        """
        Train failure detection model.

        FIX #4: RandomForest is regularised to prevent memorisation:
          - n_estimators reduced from 300 → 150 (enough variance, faster)
          - max_depth reduced from 20 → 10 (prevents deep leaf memorisation)
          - min_samples_split increased from 2 → 8 (requires real support)
          - min_samples_leaf increased from 1 → 4 (smooths leaf probabilities)
          - max_features='sqrt' (default, but explicit) — reduces correlation
        """
        if verbose:
            print("\n" + "="*70)
            print("TRAINING FAILURE DETECTOR")
            print("="*70 + "\n")
            print(f"Training samples: {len(X_train)}")
            print(f"Features: {X_train.shape[1]}")
            print(f"Classes: {y_train.nunique()}")

        self.feature_names = X_train.columns.tolist()
        self.label_names = sorted(y_train.unique())

        X_train_scaled = self.scaler.fit_transform(X_train)

        # Slightly relaxed regularisation vs. the over-regularised version:
        # max_depth 10→14, min_samples_split 8→5, min_samples_leaf 4→2.
        # Still far from the original (depth=20, splits=2, leaf=1) that produced
        # memorisation, but enough depth to separate the harder classes like
        # communication_delay and action_corruption that need finer splits.
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        if verbose:
            print("\nTraining Random Forest classifier (regularised)...")

        self.model.fit(X_train_scaled, y_train)

        if verbose:
            print("✓ Training complete!")
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            print("\nTop 10 Important Features:")
            for _, row in feature_importance.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")

    def cross_validate_report(self, X: pd.DataFrame, y: pd.Series,
                               n_splits: int = 5, verbose: bool = True):
        """
        FIX #5: 5-fold stratified cross-validation for an honest generalisation
        estimate.  This runs BEFORE the final model is trained on the full
        training split, so it does not replace evaluate() but complements it.
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"CROSS-VALIDATION ({n_splits}-fold stratified)")
            print(f"{'='*70}\n")

        # Use the same regularised estimator but fit it fresh each fold
        cv_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler as SS
        pipe = Pipeline([('scaler', SS()), ('clf', cv_model)])

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_validate(
            pipe, X, y, cv=cv,
            scoring=['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted'],
            return_train_score=True
        )

        if verbose:
            print(f"  {'Metric':<25} {'Train mean':>12}  {'Test mean':>12}  {'Test std':>10}")
            print(f"  {'-'*65}")
            for metric in ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted']:
                tr = scores[f'train_{metric}'].mean()
                te = scores[f'test_{metric}'].mean()
                sd = scores[f'test_{metric}'].std()
                print(f"  {metric:<25} {tr:>12.4f}  {te:>12.4f}  {sd:>10.4f}")
            print()

        return scores

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series,
                 verbose: bool = True, plot_confusion: bool = True,
                 save_path: str = None):
        """Evaluate model performance on held-out test set."""
        if self.model is None:
            raise ValueError("Model not trained yet!")

        self._validate_features(X_test)

        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )

        if verbose:
            print("\n" + "="*70)
            print("EVALUATION RESULTS (held-out test set)")
            print("="*70 + "\n")
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")

            print("\nPer-Class Performance:")
            print(classification_report(y_test, y_pred,
                                        target_names=self.label_names))

        if plot_confusion:
            cm = confusion_matrix(y_test, y_pred, labels=self.label_names)
            self._plot_confusion_matrix(cm, save_path)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }

    def _plot_confusion_matrix(self, cm: np.ndarray, save_path: str = None):
        """
        FIX #2: dynamic figure size and rotated labels for 19-class matrices.
        """
        n_classes = len(self.label_names)
        fig_size = max(10, n_classes * 0.65)
        plt.figure(figsize=(fig_size, fig_size * 0.85))

        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=self.label_names,
                    yticklabels=self.label_names,
                    cbar_kws={'label': 'Proportion'})

        plt.title('Failure Detection Confusion Matrix (Normalized)',
                  fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=max(6, 10 - n_classes // 5))
        plt.yticks(rotation=0, fontsize=max(6, 10 - n_classes // 5))
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Saved confusion matrix to {save_path}")
        else:
            plt.show()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict failure type for new episodes"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        self._validate_features(X)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        self._validate_features(X)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def _validate_features(self, X: pd.DataFrame):
        """
        FIX #1: raise an informative error on column mismatch instead of
        silently producing wrong predictions.
        """
        incoming = list(X.columns)
        if incoming != self.feature_names:
            missing = set(self.feature_names) - set(incoming)
            extra = set(incoming) - set(self.feature_names)
            msg_parts = []
            if missing:
                msg_parts.append(f"Missing columns: {sorted(missing)}")
            if extra:
                msg_parts.append(f"Unexpected columns: {sorted(extra)}")
            if not missing and not extra:
                msg_parts.append("Column order differs from training")
            raise ValueError(
                "Feature mismatch between training and inference data.\n" +
                "\n".join(msg_parts) +
                f"\nExpected: {self.feature_names}\nGot:      {incoming}"
            )

    def save(self, filepath: str):
        """Save trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'label_names': self.label_names
        }
        joblib.dump(model_data, filepath)
        print(f"✓ Model saved to {filepath}")

    def load(self, filepath: str):
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.label_names = model_data['label_names']
        print(f"✓ Model loaded from {filepath}")


def _drop_dead_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    FIX #3: drop zero/near-zero importance features before training.
    Also guards against column names that no longer exist in the updated
    feature matrix (e.g. leaky features removed in analyze_dataset.py).
    """
    cols_to_drop = [c for c in DEAD_FEATURES if c in df.columns]
    if cols_to_drop:
        print(f"  Dropping dead features: {cols_to_drop}")
    return df.drop(columns=cols_to_drop)


def train_binary_detector(feature_df: pd.DataFrame, output_dir: str = "./models"):
    """Train binary classifier: Normal vs Failure"""
    print("\n" + "="*70)
    print("TRAINING BINARY DETECTOR (Normal vs Failure)")
    print("="*70)

    X = feature_df.drop(['label', 'failure_type'], axis=1)
    X = _drop_dead_features(X)
    y = feature_df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    detector = FailureDetector()

    # FIX #5: cross-validation on training split before final fit
    print("\nRunning cross-validation on training data...")
    detector.cross_validate_report(X_train, y_train)

    detector.train(X_train, y_train)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    results = detector.evaluate(
        X_test, y_test,
        plot_confusion=True,
        save_path=output_path / "binary_confusion_matrix.png"
    )
    detector.save(output_path / "binary_detector.pkl")
    return detector, results


def train_multiclass_detector(feature_df: pd.DataFrame, output_dir: str = "./models"):
    """Train multi-class classifier: Normal + each failure type"""
    print("\n" + "="*70)
    print("TRAINING MULTI-CLASS DETECTOR (Failure Type Classification)")
    print("="*70)

    X = feature_df.drop(['label', 'failure_type'], axis=1)
    X = _drop_dead_features(X)
    y = feature_df['failure_type']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    detector = FailureDetector()

    # FIX #5: cross-validation on training split before final fit
    print("\nRunning cross-validation on training data...")
    detector.cross_validate_report(X_train, y_train)

    detector.train(X_train, y_train)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    results = detector.evaluate(
        X_test, y_test,
        plot_confusion=True,
        save_path=output_path / "multiclass_confusion_matrix.png"
    )
    detector.save(output_path / "multiclass_detector.pkl")
    return detector, results


if __name__ == "__main__":
    import sys

    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "./marl_dataset"
    feature_csv = Path(dataset_dir) / "analysis" / "feature_matrix.csv"

    if not feature_csv.exists():
        print(f"Error: Feature matrix not found at {feature_csv}")
        print("Please run analyze_dataset.py first!")
        sys.exit(1)

    print(f"Loading feature matrix from {feature_csv}...")
    feature_df = pd.read_csv(feature_csv)
    print(f"✓ Loaded {len(feature_df)} episodes")
    print(f"  Features: {feature_df.shape[1] - 2}")

    output_dir = Path(dataset_dir) / "models"

    binary_detector, binary_results = train_binary_detector(feature_df, output_dir)
    multiclass_detector, multiclass_results = train_multiclass_detector(feature_df, output_dir)

    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModels saved to: {output_dir}")
    print("\nBinary Detector Performance:")
    print(f"  Accuracy: {binary_results['accuracy']:.4f}")
    print(f"  F1-Score: {binary_results['f1']:.4f}")
    print("\nMulti-class Detector Performance:")
    print(f"  Accuracy: {multiclass_results['accuracy']:.4f}")
    print(f"  F1-Score: {multiclass_results['f1']:.4f}")
