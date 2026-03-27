import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


class FailureDetector:
    """
    Baseline failure detection model
    Classifies episodes as normal or failure type
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = None
        self.label_names = None
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              verbose: bool = True):
        """
        Train failure detection model
        
        Args:
            X_train: Feature matrix (episodes x features)
            y_train: Labels (failure types or 'normal')
            verbose: Print training progress
        """
        if verbose:
            print("\n" + "="*70)
            print("TRAINING FAILURE DETECTOR")
            print("="*70 + "\n")
            print(f"Training samples: {len(X_train)}")
            print(f"Features: {X_train.shape[1]}")
            print(f"Classes: {y_train.nunique()}")
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        self.label_names = sorted(y_train.unique())
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train Random Forest classifier with balanced weights
        self.model = RandomForestClassifier(
            n_estimators=300,          # Increased for 19 classes
            max_depth=20,             # Slightly deeper for more complex features
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        if verbose:
            print("\nTraining Random Forest classifier...")
        
        self.model.fit(X_train_scaled, y_train)
        
        if verbose:
            print("✓ Training complete!")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 5 Important Features:")
            for idx, row in feature_importance.head().iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series,
                 verbose: bool = True, plot_confusion: bool = True,
                 save_path: str = None):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            verbose: Print metrics
            plot_confusion: Show confusion matrix
            save_path: Path to save confusion matrix plot
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Scale test features
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        if verbose:
            print("\n" + "="*70)
            print("EVALUATION RESULTS")
            print("="*70 + "\n")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            
            print("\nPer-Class Performance:")
            print(classification_report(y_test, y_pred, 
                                       target_names=self.label_names))
        
        # Confusion matrix
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
        """Plot confusion matrix heatmap"""
        plt.figure(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=self.label_names,
                   yticklabels=self.label_names,
                   cbar_kws={'label': 'Proportion'})
        
        plt.title('Failure Detection Confusion Matrix (Normalized)', 
                 fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
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
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
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


def train_binary_detector(feature_df: pd.DataFrame, output_dir: str = "./models"):
    """
    Train binary classifier: Normal vs Failure
    """
    print("\n" + "="*70)
    print("TRAINING BINARY DETECTOR (Normal vs Failure)")
    print("="*70)
    
    # Prepare data
    X = feature_df.drop(['label', 'failure_type'], axis=1)
    y = feature_df['label']  # 'normal' or 'failure'
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train
    detector = FailureDetector()
    detector.train(X_train, y_train)
    
    # Evaluate
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    results = detector.evaluate(
        X_test, y_test,
        plot_confusion=True,
        save_path=output_path / "binary_confusion_matrix.png"
    )
    
    # Save model
    detector.save(output_path / "binary_detector.pkl")
    
    return detector, results


def train_multiclass_detector(feature_df: pd.DataFrame, output_dir: str = "./models"):
    """
    Train multi-class classifier: Normal + each failure type
    """
    print("\n" + "="*70)
    print("TRAINING MULTI-CLASS DETECTOR (Failure Type Classification)")
    print("="*70)
    
    # Prepare data - use failure_type as label (includes 'none' for normal)
    X = feature_df.drop(['label', 'failure_type'], axis=1)
    y = feature_df['failure_type']  # 'none', 'communication_delay', etc.
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train
    detector = FailureDetector()
    detector.train(X_train, y_train)
    
    # Evaluate
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    results = detector.evaluate(
        X_test, y_test,
        plot_confusion=True,
        save_path=output_path / "multiclass_confusion_matrix.png"
    )
    
    # Save model
    detector.save(output_path / "multiclass_detector.pkl")
    
    return detector, results


if __name__ == "__main__":
    import sys
    
    # Usage: python train_detector.py [dataset_dir]
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "./marl_dataset"
    
    # Load feature matrix
    feature_csv = Path(dataset_dir) / "analysis" / "feature_matrix.csv"
    
    if not feature_csv.exists():
        print(f"Error: Feature matrix not found at {feature_csv}")
        print("Please run analyze_dataset.py first!")
        sys.exit(1)
    
    print(f"Loading feature matrix from {feature_csv}...")
    feature_df = pd.read_csv(feature_csv)
    print(f"✓ Loaded {len(feature_df)} episodes")
    print(f"  Features: {feature_df.shape[1] - 2}")  # -2 for label columns
    
    # Create output directory
    output_dir = Path(dataset_dir) / "models"
    
    # Train binary detector
    binary_detector, binary_results = train_binary_detector(feature_df, output_dir)
    
    # Train multi-class detector
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
