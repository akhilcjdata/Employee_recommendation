"""
Model Training Pipeline for ML Recommendation System
Takes feature-engineered data and trains multiple ML models

Input: data/features/latest_features.csv + models/embeddings.pkl
Output: Trained models in models/ folder
"""

import pandas as pd
import numpy as np
import joblib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import warnings

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """ML Model training for shift recommendation system"""
    
    def __init__(self, 
                 features_data_path: str = "D:/Employee_Suggestions/employee_suggestion_system/data/interim",
                 models_output_path: str = "D:/Employee_Suggestions/employee_suggestion_system/models"):
        
        self.features_data_path = Path(features_data_path)
        self.models_output_path = Path(models_output_path)
        
        # Create models directory
        self.models_output_path.mkdir(parents=True, exist_ok=True)
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.target_columns = ['assignment_success', 'employee_satisfaction', 'employer_satisfaction', 'completion_likelihood']
        
        # Training results
        self.training_results = {}
        self.model_metadata = {}
        
        logger.info(f"Features data path: {self.features_data_path}")
        logger.info(f"Models output path: {self.models_output_path}")
    
    def load_training_data(self, filename: str = "latest_features.csv") -> pd.DataFrame:
        """
        Load feature-engineered data for training
        """
        
        file_path = self.features_data_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Features file not found: {file_path}")
        
        logger.info(f"Loading training data from: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            
            # Load feature metadata
            metadata_path = self.features_data_path / "feature_metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    feature_metadata = json.load(f)
                    self.feature_columns = feature_metadata.get('feature_columns', [])
            
            # If no metadata, extract feature columns automatically
            if not self.feature_columns:
                exclude_cols = [
                    'EmployeeId', 'SiteId', 'TaskId', 'EmployerId', 'CreatedBy', 
                    'ScheduleWeek', 'Week', 'FKshiftId', 'processed_at',
                    'feature_engineering_timestamp'
                ] + self.target_columns
                
                self.feature_columns = [col for col in df.columns if col not in exclude_cols]
            
            logger.info(f"✅ Training data loaded successfully!")
            logger.info(f"   📊 Shape: {df.shape}")
            logger.info(f"   🎯 Features: {len(self.feature_columns)}")
            logger.info(f"   📋 Targets: {len(self.target_columns)}")
            logger.info(f"   📅 Date range: {df['ScheduleWeek'].min()} to {df['ScheduleWeek'].max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to load training data: {str(e)}")
            raise
    
    def load_embeddings(self) -> Dict:
        """
        Load pre-trained embeddings
        """
        
        embeddings_path = self.models_output_path / "embeddings.pkl"
        
        if not embeddings_path.exists():
            logger.warning(f"⚠️ Embeddings file not found: {embeddings_path}")
            return {}
        
        try:
            embeddings = joblib.load(embeddings_path)
            logger.info(f"✅ Embeddings loaded from: {embeddings_path}")
            return embeddings
        except Exception as e:
            logger.error(f"❌ Failed to load embeddings: {str(e)}")
            return {}
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Prepare feature matrix and target variables for training
        """
        
        logger.info("🔧 Preparing training data...")
        
        # Check if all feature columns exist
        missing_features = [col for col in self.feature_columns if col not in df.columns]
        if missing_features:
            logger.warning(f"⚠️ Missing feature columns: {missing_features}")
            self.feature_columns = [col for col in self.feature_columns if col in df.columns]
        
        # Check if all target columns exist
        missing_targets = [col for col in self.target_columns if col not in df.columns]
        if missing_targets:
            raise ValueError(f"Missing target columns: {missing_targets}")
        
        # Extract features
        X = df[self.feature_columns].copy()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Extract targets
        y = {}
        for target in self.target_columns:
            y[target] = df[target].values
        
        # Scale features
        logger.info("   📏 Scaling features...")
        self.scalers['main'] = StandardScaler()
        X_scaled = self.scalers['main'].fit_transform(X)
        
        logger.info(f"   ✅ Data prepared: {X_scaled.shape[0]:,} samples, {X_scaled.shape[1]} features")
        
        return X_scaled, y
    
    def train_success_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Train assignment success prediction model
        """
        
        logger.info("📊 Training Success Prediction Model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        # Try different models and pick the best
        models_to_try = {
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                subsample=0.8
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=150,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )
        }
        
        best_model = None
        best_score = -np.inf
        
        for name, model in models_to_try.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            avg_score = cv_scores.mean()
            
            logger.info(f"   {name}: CV R² = {avg_score:.3f} ± {cv_scores.std():.3f}")
            
            if avg_score > best_score:
                best_score = avg_score
                best_model = model
        
        # Train best model
        best_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = best_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # Store model
        self.models['success'] = best_model
        
        results = {
            'r2_score': r2,
            'rmse': rmse,
            'mae': mae,
            'cv_score': best_score
        }
        
        logger.info(f"   ✅ Success Model: R² = {r2:.3f}, RMSE = {rmse:.3f}")
        
        return results
    
    def train_satisfaction_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Train employee satisfaction prediction model
        """
        
        logger.info("😊 Training Employee Satisfaction Model...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Random Forest works well for satisfaction scores
        model = RandomForestRegressor(
            n_estimators=150,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # Store model
        self.models['satisfaction'] = model
        
        results = {
            'r2_score': r2,
            'rmse': rmse,
            'mae': mae
        }
        
        logger.info(f"   ✅ Satisfaction Model: R² = {r2:.3f}, RMSE = {rmse:.3f}")
        
        return results
    
    def train_quality_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Train employer satisfaction/quality prediction model
        """
        
        logger.info("⭐ Training Employer Satisfaction Model...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Gradient Boosting for quality prediction
        model = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            random_state=42
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # Store model
        self.models['quality'] = model
        
        results = {
            'r2_score': r2,
            'rmse': rmse,
            'mae': mae
        }
        
        logger.info(f"   ✅ Quality Model: R² = {r2:.3f}, RMSE = {rmse:.3f}")
        
        return results
    
    def train_completion_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Train completion likelihood prediction model
        """
        
        logger.info("✅ Training Completion Likelihood Model...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Random Forest for completion prediction
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        
        # Train model
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        # Store model
        self.models['completion'] = model
        
        results = {
            'r2_score': r2,
            'rmse': rmse,
            'mae': mae
        }
        
        logger.info(f"   ✅ Completion Model: R² = {r2:.3f}, RMSE = {rmse:.3f}")
        
        return results
    
    def train_neural_ensemble(self, X: np.ndarray, y_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Train deep learning ensemble model for multi-output prediction
        """
        
        logger.info("🧠 Training Neural Network Ensemble...")
        
        # Prepare multi-output targets
        y_multi = np.column_stack([y_dict[target] for target in self.target_columns])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_multi, test_size=0.2, random_state=42
        )
        
        # Build neural network architecture
        input_dim = X.shape[1]
        
        model = keras.Sequential([
            # Input layer
            layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # Hidden layers
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.1),
            
            layers.Dense(32, activation='relu'),
            
            # Output layer (4 targets)
            layers.Dense(len(self.target_columns), activation='linear')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=0.00001,
            verbose=1
        )
        
        # Train model
        logger.info("   🔥 Training neural network...")
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=64,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate
        y_pred = model.predict(X_test, verbose=0)
        
        # Calculate metrics for each output
        results = {}
        for i, target in enumerate(self.target_columns):
            r2 = r2_score(y_test[:, i], y_pred[:, i])
            rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
            mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
            
            results[f'{target}_r2'] = r2
            results[f'{target}_rmse'] = rmse
            results[f'{target}_mae'] = mae
            
            logger.info(f"   {target}: R² = {r2:.3f}, RMSE = {rmse:.3f}")
        
        # Overall metrics
        val_loss = model.evaluate(X_test, y_test, verbose=0)
        results['overall_loss'] = val_loss[0]
        results['overall_mae'] = val_loss[1]
        
        # Store model
        self.models['neural_ensemble'] = model
        
        logger.info(f"   ✅ Neural Ensemble: Loss = {val_loss[0]:.4f}, MAE = {val_loss[1]:.4f}")
        
        return results
    
    def evaluate_all_models(self, X: np.ndarray, y_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Comprehensive evaluation of all trained models
        """
        
        logger.info("📈 Evaluating all models...")
        
        # Create test set
        X_train, X_test, _, _ = train_test_split(X, list(y_dict.values())[0], test_size=0.2, random_state=42)
        
        evaluation_results = {}
        
        # Evaluate individual models
        for target in self.target_columns:
            if target.replace('_', '') in ['assignmentsuccess', 'success']:
                model_key = 'success'
            elif 'satisfaction' in target and 'employee' in target:
                model_key = 'satisfaction'
            elif 'satisfaction' in target and 'employer' in target:
                model_key = 'quality'
            elif 'completion' in target:
                model_key = 'completion'
            else:
                continue
            
            if model_key in self.models:
                _, y_test_target = train_test_split(y_dict[target], test_size=0.2, random_state=42)
                y_pred = self.models[model_key].predict(X_test)
                
                evaluation_results[f'{model_key}_model'] = {
                    'target': target,
                    'r2_score': r2_score(y_test_target, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test_target, y_pred)),
                    'mae': mean_absolute_error(y_test_target, y_pred)
                }
        
        # Feature importance analysis
        if 'success' in self.models:
            feature_importance = {}
            if hasattr(self.models['success'], 'feature_importances_'):
                importances = self.models['success'].feature_importances_
                # Get top 10 features
                top_indices = np.argsort(importances)[-10:][::-1]
                for idx in top_indices:
                    if idx < len(self.feature_columns):
                        feature_importance[self.feature_columns[idx]] = float(importances[idx])
                
                evaluation_results['feature_importance'] = feature_importance
        
        return evaluation_results
    
    def save_models(self) -> str:
        """
        Save all trained models and metadata
        """
        
        logger.info("💾 Saving trained models...")
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            # Save individual models
            for model_name, model in self.models.items():
                if model_name == 'neural_ensemble':
                    # Save Keras model
                    model_path = self.models_output_path / f"{model_name}_{timestamp}.h5"
                    model.save(model_path)
                    
                    # Also save as latest
                    latest_path = self.models_output_path / f"{model_name}_latest.h5"
                    model.save(latest_path)
                else:
                    # Save sklearn models
                    model_path = self.models_output_path / f"{model_name}_model_{timestamp}.pkl"
                    joblib.dump(model, model_path)
                    
                    # Also save as latest
                    latest_path = self.models_output_path / f"{model_name}_model_latest.pkl"
                    joblib.dump(model, latest_path)
                
                logger.info(f"   ✅ Saved {model_name} model")
            
            # Save scalers
            scalers_path = self.models_output_path / f"scalers_{timestamp}.pkl"
            joblib.dump(self.scalers, scalers_path)
            
            latest_scalers_path = self.models_output_path / "scalers_latest.pkl"
            joblib.dump(self.scalers, latest_scalers_path)
            
            # Save complete model package
            model_package = {
                'models_info': {name: type(model).__name__ for name, model in self.models.items()},
                'feature_columns': self.feature_columns,
                'target_columns': self.target_columns,
                'training_results': self.training_results,
                'metadata': {
                    'training_timestamp': timestamp,
                    'training_date': datetime.now().isoformat(),
                    'total_models': len(self.models),
                    'total_features': len(self.feature_columns),
                    'model_version': '1.0'
                }
            }
            
            # Save metadata
            metadata_path = self.models_output_path / f"model_metadata_{timestamp}.json"
            with open(metadata_path, 'w') as f:
                json.dump(model_package, f, indent=2, default=str)
            
            latest_metadata_path = self.models_output_path / "model_metadata_latest.json"
            with open(latest_metadata_path, 'w') as f:
                json.dump(model_package, f, indent=2, default=str)
            
            logger.info(f"✅ All models saved successfully!")
            logger.info(f"   📁 Models: {self.models_output_path}")
            logger.info(f"   📊 {len(self.models)} models trained and saved")
            logger.info(f"   📋 Metadata: {metadata_path}")
            
            return str(self.models_output_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to save models: {str(e)}")
            raise
    
    def run_training_pipeline(self, input_filename: str = "latest_features.csv") -> Dict[str, Any]:
        """
        Complete model training pipeline
        """
        
        logger.info("🚀 Starting model training pipeline...")
        start_time = datetime.now()
        
        try:
            # Step 1: Load training data
            df = self.load_training_data(input_filename)
            
            # Step 2: Load embeddings (optional)
            embeddings = self.load_embeddings()
            
            # Step 3: Prepare training data
            X, y = self.prepare_training_data(df)
            
            # Step 4: Train individual models
            logger.info("🎯 Training individual models...")
            
            success_results = self.train_success_model(X, y['assignment_success'])
            satisfaction_results = self.train_satisfaction_model(X, y['employee_satisfaction'])
            quality_results = self.train_quality_model(X, y['employer_satisfaction'])
            completion_results = self.train_completion_model(X, y['completion_likelihood'])
            
            # Step 5: Train neural ensemble
            neural_results = self.train_neural_ensemble(X, y)
            
            # Step 6: Evaluate all models
            evaluation_results = self.evaluate_all_models(X, y)
            
            # Store training results
            self.training_results = {
                'success_model': success_results,
                'satisfaction_model': satisfaction_results,
                'quality_model': quality_results,
                'completion_model': completion_results,
                'neural_ensemble': neural_results,
                'evaluation': evaluation_results
            }
            
            # Step 7: Save all models
            models_path = self.save_models()
            
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            
            result = {
                'status': 'success',
                'input_file': input_filename,
                'models_saved_to': models_path,
                'models_trained': list(self.models.keys()),
                'training_results': self.training_results,
                'duration_seconds': round(duration, 2),
                'summary': {
                    'total_models': len(self.models),
                    'total_features': len(self.feature_columns),
                    'total_samples': X.shape[0],
                    'best_model_r2': max([
                        success_results.get('r2_score', 0),
                        satisfaction_results.get('r2_score', 0),
                        quality_results.get('r2_score', 0),
                        completion_results.get('r2_score', 0)
                    ])
                }
            }
            
            logger.info("🎉 Model training completed successfully!")
            logger.info(f"   ⏱️  Duration: {duration:.2f} seconds")
            logger.info(f"   🤖 Models trained: {len(self.models)}")
            logger.info(f"   📊 Best R² score: {result['summary']['best_model_r2']:.3f}")
            logger.info(f"   📁 Saved to: {models_path}")
            
            return result
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            
            logger.error(f"❌ Model training failed after {duration:.2f} seconds")
            logger.error(f"   Error: {str(e)}")
            
            return {
                'status': 'failed',
                'error': str(e),
                'duration_seconds': round(duration, 2)
            }

def main():
    """Main function for running model training"""
    
    print("🤖 ML Model Training Pipeline")
    print("=" * 50)
    
    # Check for command line arguments
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h']:
            print("Usage:")
            print("  python model_training.py                    # Use latest features")
            print("  python model_training.py features_file.csv  # Specify features file")
            return
        
        input_file = sys.argv[1]
    else:
        input_file = "latest_features.csv"
    
    # Initialize model trainer
    trainer = ModelTrainer()
    
    # Run training pipeline
    result = trainer.run_training_pipeline(input_file)
    
    # Print results
    if result['status'] == 'success':
        print(f"\n✅ SUCCESS!")
        print(f"🤖 Trained {result['summary']['total_models']} models")
        print(f"📊 Features used: {result['summary']['total_features']}")
        print(f"📈 Samples trained on: {result['summary']['total_samples']:,}")
        print(f"🎯 Best R² score: {result['summary']['best_model_r2']:.3f}")
        print(f"\n📊 Individual Model Performance:")
        
        # Show individual model results
        for model_name, model_results in result['training_results'].items():
            if isinstance(model_results, dict) and 'r2_score' in model_results:
                print(f"   {model_name}: R² = {model_results['r2_score']:.3f}, "
                      f"RMSE = {model_results.get('rmse', 0):.3f}")
        
        print(f"\n💾 Models saved to: {result['models_saved_to']}")
        print(f"⏱️ Training time: {result['duration_seconds']:.2f} seconds")
        
    else:
        print(f"\n❌ FAILED!")
        print(f"Error: {result['error']}")
    
    return result

if __name__ == "__main__":
    result = main()