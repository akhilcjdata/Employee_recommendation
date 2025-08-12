"""
Feature Engineering Pipeline for ML Recommendation System
Takes cleaned data and creates ML-ready features

Input: data/processed/processed_employee_schedule_data_*.csv
Output: data/features/feature_engineered_data.csv
"""

import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import warnings

# ML Libraries for feature engineering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureEngineering:
    """Feature engineering for shift recommendation ML system"""
    
    def __init__(self, 
                 processed_data_path: str = "D:/Employee_Suggestions/employee_suggestion_system/data/processed", 
                 features_data_path: str = "D:/Employee_Suggestions/employee_suggestion_system/data/interim"):
        
        self.processed_data_path = Path(processed_data_path)
        self.features_data_path = Path(features_data_path)
        
        # Create features directory
        self.features_data_path.mkdir(parents=True, exist_ok=True)
        
        # Feature engineering components
        self.employee_embeddings = {}
        self.encoders = {}
        self.feature_columns = []
        
        logger.info(f"Processed data path: {self.processed_data_path}")
        logger.info(f"Features output path: {self.features_data_path}")
    
    def load_processed_data(self, filename: str = None) -> pd.DataFrame:
        """
        Load the processed/cleaned data for feature engineering
        
        Args:
            filename: Specific file to load, if None uses latest
            
        Returns:
            pandas.DataFrame: Cleaned data ready for feature engineering
        """
        
        if filename is None:
            # Use the latest processed data file
            latest_file = self.processed_data_path / "latest_processed_data.csv"
            if latest_file.exists():
                file_path = latest_file
            else:
                # Find the most recent processed file
                processed_files = list(self.processed_data_path.glob("processed_employee_schedule_data_*.csv"))
                if not processed_files:
                    raise FileNotFoundError(f"No processed data files found in {self.processed_data_path}")
                file_path = max(processed_files, key=os.path.getctime)
        else:
            file_path = self.processed_data_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Processed data file not found: {file_path}")
        
        logger.info(f"Loading processed data from: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            
            logger.info(f"✅ Processed data loaded successfully!")
            logger.info(f"   📊 Shape: {df.shape}")
            logger.info(f"   📋 Columns: {list(df.columns)}")
            logger.info(f"   📅 Date range: {df['ScheduleWeek'].min()} to {df['ScheduleWeek'].max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to load processed data: {str(e)}")
            raise
    
    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables for ML training
        In real system, these would come from actual feedback data
        For now, we simulate realistic targets based on patterns
        """
        
        logger.info("🎯 Creating target variables for ML training...")
        
        np.random.seed(42)  # For reproducible results
        
        # Simulate assignment success (0-1 probability)
        logger.info("   Creating assignment_success target...")
        
        # Higher success for experienced employees and familiar combinations
        employee_experience = df.groupby('EmployeeId')['EmployeeId'].transform('count') / 100
        
        # Site familiarity - employees who worked at this site before
        site_experience = df.groupby(['EmployeeId', 'SiteId']).cumcount() + 1
        site_familiarity = np.log(site_experience) / 3  # Logarithmic benefit
        
        # Base success rate with realistic factors
        base_success = 0.85
        df['assignment_success'] = np.clip(
            base_success + 
            (employee_experience * 0.1) + 
            (site_familiarity * 0.05) + 
            np.random.normal(0, 0.08, len(df)), 0, 1
        )
        
        # Simulate employee satisfaction (1-10 scale)
        logger.info("   Creating employee_satisfaction target...")
        
        # Higher satisfaction for balanced workload and familiar environments
        workload_balance = 1 - abs(employee_experience - 0.5)  # Optimal around 0.5
        
        df['employee_satisfaction'] = np.clip(
            7.5 + 
            (workload_balance * 2) + 
            (site_familiarity * 0.5) + 
            np.random.normal(0, 0.8, len(df)), 1, 10
        )
        
        # Simulate employer satisfaction (1-10 scale)
        logger.info("   Creating employer_satisfaction target...")
        
        # Higher employer satisfaction for experienced employees
        df['employer_satisfaction'] = np.clip(
            8.0 + 
            (employee_experience * 1.5) + 
            (site_familiarity * 0.8) + 
            np.random.normal(0, 0.6, len(df)), 1, 10
        )
        
        # Simulate completion likelihood (0-1 probability)
        logger.info("   Creating completion_likelihood target...")
        
        # Higher completion for consistent, experienced employees
        df['completion_likelihood'] = np.clip(
            0.92 + 
            (employee_experience * 0.06) + 
            (site_familiarity * 0.02) + 
            np.random.normal(0, 0.04, len(df)), 0, 1
        )
        
        logger.info(f"✅ Target variables created!")
        logger.info(f"   📊 Assignment success avg: {df['assignment_success'].mean():.3f}")
        logger.info(f"   😊 Employee satisfaction avg: {df['employee_satisfaction'].mean():.1f}")
        logger.info(f"   ⭐ Employer satisfaction avg: {df['employer_satisfaction'].mean():.1f}")
        logger.info(f"   ✅ Completion likelihood avg: {df['completion_likelihood'].mean():.3f}")
        
        return df
    
    def engineer_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from schedule data
        """
        
        logger.info("📅 Engineering temporal features...")
        
        # Extract year and week components if not already present
        if 'Schedule_Year' not in df.columns:
            df['Schedule_Year'] = df['ScheduleWeek'].astype(str).str[:4].astype(int)
        if 'Schedule_WeekNumber' not in df.columns:
            df['Schedule_WeekNumber'] = df['ScheduleWeek'].astype(str).str[4:].astype(int)
        
        # Day of year calculation
        try:
            df['DayOfYear'] = pd.to_datetime(
                df['ScheduleWeek'].astype(str) + '1', 
                format='%Y%W%w',
                errors='coerce'
            ).dt.dayofyear
            
            # Fill any NaT values with approximation
            df['DayOfYear'] = df['DayOfYear'].fillna(
                (df['Schedule_WeekNumber'] - 1) * 7
            )
        except:
            # Fallback calculation
            df['DayOfYear'] = (df['Schedule_WeekNumber'] - 1) * 7
        
        # Weekend indicator
        df['IsWeekend'] = df['WeekDay'].isin([6, 7]).astype(int)
        
        # Season calculation (0=Winter, 1=Spring, 2=Summer, 3=Fall)
        df['Season'] = ((df['DayOfYear'] - 1) // 91).clip(0, 3).astype(int)
        
        # Month approximation
        df['Month'] = ((df['DayOfYear'] - 1) // 30.44).clip(0, 11).astype(int) + 1
        
        # Quarter
        df['Quarter'] = ((df['Month'] - 1) // 3 + 1).astype(int)
        
        # Holiday periods (simplified)
        df['IsHolidayPeriod'] = (
            (df['Month'].isin([12, 1])) |  # Winter holidays
            (df['Month'] == 7) |           # Summer vacation
            (df['Month'] == 4)             # Spring break
        ).astype(int)
        
        logger.info(f"   ✅ Created temporal features: Season, Month, Quarter, Holiday periods")
        
        return df
    
    def engineer_employee_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create employee-level aggregated features
        """
        
        logger.info("👥 Engineering employee-level features...")
        
        # Calculate employee statistics
        employee_stats = df.groupby('EmployeeId').agg({
            'SiteId': ['count', 'nunique'],
            'TaskId': 'nunique',
            'EmployerId': 'nunique',
            'WeekDay': [lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.mean(), 'std'],
            'Schedule_Year': ['min', 'max'],
            'Schedule_WeekNumber': ['std', 'mean', 'min', 'max'],
            'IsWeekend': 'mean'
        }).round(3)
        
        # Flatten column names
        employee_stats.columns = [f'emp_{col[0]}_{col[1]}' for col in employee_stats.columns]
        
        # Calculate derived metrics
        employee_stats['emp_years_active'] = (
            employee_stats['emp_Schedule_Year_max'] - 
            employee_stats['emp_Schedule_Year_min'] + 1
        )
        
        employee_stats['emp_weekly_consistency'] = 1 / (
            employee_stats['emp_Schedule_WeekNumber_std'].fillna(1) + 1
        )
        
        employee_stats['emp_weekend_preference'] = employee_stats['emp_IsWeekend_mean']
        
        # Versatility metrics
        employee_stats['emp_site_versatility'] = (
            employee_stats['emp_SiteId_nunique'] / employee_stats['emp_SiteId_count']
        )
        
        employee_stats['emp_task_versatility'] = (
            employee_stats['emp_TaskId_nunique'] / employee_stats['emp_SiteId_count']
        )
        
        employee_stats['emp_employer_versatility'] = (
            employee_stats['emp_EmployerId_nunique'] / employee_stats['emp_SiteId_count']
        )
        
        # Merge back to main dataframe
        df = df.merge(employee_stats, left_on='EmployeeId', right_index=True, how='left')
        
        logger.info(f"   ✅ Created {len(employee_stats.columns)} employee-level features")
        
        return df
    
    def engineer_site_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create site-level aggregated features
        """
        
        logger.info("📍 Engineering site-level features...")
        
        # Site statistics
        site_stats = df.groupby('SiteId').agg({
            'EmployeeId': 'nunique',
            'TaskId': 'nunique',
            'EmployerId': 'nunique',
            'assignment_success': 'mean',
            'employee_satisfaction': 'mean',
            'completion_likelihood': 'mean',
            'WeekDay': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.mean(),
            'IsWeekend': 'mean'
        }).round(3)
        
        site_stats.columns = [f'site_{col}' for col in site_stats.columns]
        
        # Site complexity and attractiveness
        site_stats['site_complexity'] = site_stats['site_TaskId']
        site_stats['site_popularity'] = site_stats['site_EmployeeId']
        site_stats['site_weekend_activity'] = site_stats['site_IsWeekend']
        
        # Merge back to main dataframe
        df = df.merge(site_stats, left_on='SiteId', right_index=True, how='left')
        
        logger.info(f"   ✅ Created {len(site_stats.columns)} site-level features")
        
        return df
    
    def engineer_task_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create task-level aggregated features
        """
        
        logger.info("🛠️ Engineering task-level features...")
        
        # Task statistics
        task_stats = df.groupby('TaskId').agg({
            'EmployeeId': 'nunique',
            'SiteId': 'nunique',
            'assignment_success': 'mean',
            'completion_likelihood': 'mean',
            'employer_satisfaction': 'mean',
            'WeekDay': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.mean(),
        }).round(3)
        
        task_stats.columns = [f'task_{col}' for col in task_stats.columns]
        
        # Task difficulty and demand
        task_stats['task_difficulty'] = 1 - task_stats['task_assignment_success']
        task_stats['task_demand'] = task_stats['task_EmployeeId']
        task_stats['task_specialization'] = 1 / (task_stats['task_SiteId'] + 1)
        
        # Merge back to main dataframe
        df = df.merge(task_stats, left_on='TaskId', right_index=True, how='left')
        
        logger.info(f"   ✅ Created {len(task_stats.columns)} task-level features")
        
        return df
    
    def engineer_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between employee, site, task combinations
        """
        
        logger.info("🔗 Engineering interaction features...")
        
        # Employee-Site experience
        df['emp_site_experience'] = df.groupby(['EmployeeId', 'SiteId']).cumcount() + 1
        df['emp_site_familiarity'] = np.log(df['emp_site_experience']) / 3
        
        # Employee-Task experience
        df['emp_task_experience'] = df.groupby(['EmployeeId', 'TaskId']).cumcount() + 1
        df['emp_task_familiarity'] = np.log(df['emp_task_experience']) / 3
        
        # Employee-Employer experience
        df['emp_employer_experience'] = df.groupby(['EmployeeId', 'EmployerId']).cumcount() + 1
        df['emp_employer_familiarity'] = np.log(df['emp_employer_experience']) / 3
        
        # Site-Task combination experience
        site_task_stats = df.groupby(['SiteId', 'TaskId']).agg({
            'EmployeeId': 'nunique',
            'assignment_success': 'mean'
        }).round(3)
        site_task_stats.columns = ['site_task_employee_count', 'site_task_success_rate']
        df = df.merge(site_task_stats, left_on=['SiteId', 'TaskId'], right_index=True, how='left')
        
        # Recency features
        current_week = df['ScheduleWeek'].max()
        df['weeks_since_assignment'] = current_week - df.groupby('EmployeeId')['ScheduleWeek'].transform('max')
        df['is_recent_worker'] = (df['weeks_since_assignment'] <= 12).astype(int)
        df['is_very_recent_worker'] = (df['weeks_since_assignment'] <= 4).astype(int)
        
        logger.info(f"   ✅ Created interaction and recency features")
        
        return df
    
    def engineer_workload_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create workload balance and fairness features
        """
        
        logger.info("⚖️ Engineering workload and fairness features...")
        
        # Calculate average assignments across all employees
        avg_assignments = df.groupby('EmployeeId')['EmployeeId'].transform('count').mean()
        
        # Individual workload metrics
        df['workload_count'] = df.groupby('EmployeeId')['EmployeeId'].transform('count')
        df['workload_ratio'] = df['workload_count'] / avg_assignments
        
        # Opportunity deficit (how far below average)
        df['opportunity_deficit'] = np.maximum(0, avg_assignments - df['workload_count'])
        df['opportunity_surplus'] = np.maximum(0, df['workload_count'] - avg_assignments)
        
        # Fairness tiers
        df['needs_opportunities'] = (df['workload_ratio'] < 0.7).astype(int)
        df['has_many_opportunities'] = (df['workload_ratio'] > 1.5).astype(int)
        
        # Workload distribution within time periods
        df['workload_recent_weeks'] = df.groupby('EmployeeId')['ScheduleWeek'].transform(
            lambda x: (x >= x.max() - 400).sum()  # Last ~8 weeks
        )
        
        logger.info(f"   ✅ Created workload balance features")
        logger.info(f"   📊 Avg assignments per employee: {avg_assignments:.1f}")
        logger.info(f"   🔴 Employees needing opportunities: {df['needs_opportunities'].mean():.1%}")
        
        return df
    
    def create_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create embedding features for categorical variables using TF-IDF
        """
        
        logger.info("🧠 Creating embedding features...")
        
        # Employee work pattern embeddings
        logger.info("   Creating employee embeddings...")
        
        # Create employee profiles based on work patterns
        employee_profiles = df.groupby('EmployeeId').agg({
            'SiteId': lambda x: ' '.join(map(str, x.value_counts().head(10).index)),
            'TaskId': lambda x: ' '.join(map(str, x.value_counts().head(10).index)),
            'WeekDay': lambda x: ' '.join(map(str, x.value_counts().head(7).index)),
            'EmployerId': lambda x: ' '.join(map(str, x.value_counts().head(5).index))
        })
        
        # Create TF-IDF embeddings for different aspects
        
        # Site preference embeddings
        site_vectorizer = TfidfVectorizer(max_features=20, token_pattern=r'\b\w+\b')
        employee_site_embeddings = site_vectorizer.fit_transform(employee_profiles['SiteId']).toarray()
        
        # Task preference embeddings  
        task_vectorizer = TfidfVectorizer(max_features=15, token_pattern=r'\b\w+\b')
        employee_task_embeddings = task_vectorizer.fit_transform(employee_profiles['TaskId']).toarray()
        
        # Weekday preference embeddings
        weekday_vectorizer = TfidfVectorizer(max_features=7, token_pattern=r'\b\w+\b')
        employee_weekday_embeddings = weekday_vectorizer.fit_transform(employee_profiles['WeekDay']).toarray()
        
        # Store embeddings for later use
        self.employee_embeddings = {
            'site_vectorizer': site_vectorizer,
            'task_vectorizer': task_vectorizer,
            'weekday_vectorizer': weekday_vectorizer,
            'site_embeddings': dict(zip(employee_profiles.index, employee_site_embeddings)),
            'task_embeddings': dict(zip(employee_profiles.index, employee_task_embeddings)),
            'weekday_embeddings': dict(zip(employee_profiles.index, employee_weekday_embeddings))
        }
        
        # Add embedding features to dataframe
        site_embed_cols = [f'emp_site_embed_{i}' for i in range(20)]
        task_embed_cols = [f'emp_task_embed_{i}' for i in range(15)]
        weekday_embed_cols = [f'emp_weekday_embed_{i}' for i in range(7)]
        
        # Initialize embedding columns
        for col in site_embed_cols + task_embed_cols + weekday_embed_cols:
            df[col] = 0.0
        
        # Fill embedding values for each employee
        for emp_id in employee_profiles.index:
            mask = df['EmployeeId'] == emp_id
            
            if emp_id in self.employee_embeddings['site_embeddings']:
                # Site embeddings
                site_values = self.employee_embeddings['site_embeddings'][emp_id]
                for i, col in enumerate(site_embed_cols):
                    df.loc[mask, col] = site_values[i]
                
                # Task embeddings
                task_values = self.employee_embeddings['task_embeddings'][emp_id]
                for i, col in enumerate(task_embed_cols):
                    df.loc[mask, col] = task_values[i]
                
                # Weekday embeddings
                weekday_values = self.employee_embeddings['weekday_embeddings'][emp_id]
                for i, col in enumerate(weekday_embed_cols):
                    df.loc[mask, col] = weekday_values[i]
        
        logger.info(f"   ✅ Created embeddings: 20 site + 15 task + 7 weekday = 42 features")
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical variables for ML
        """
        
        logger.info("🔤 Encoding categorical features...")
        
        # Categorical columns to encode
        categorical_cols = ['WeekDay', 'Season', 'Month', 'Quarter']
        
        for col in categorical_cols:
            if col in df.columns:
                # Create label encoder
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
                
                logger.info(f"   ✅ Encoded {col}: {df[col].nunique()} unique values")
        
        return df
    
    def finalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Final feature preparation and selection
        """
        
        logger.info("🎯 Finalizing features for ML...")
        
        # Identify feature columns (exclude IDs and targets)
        exclude_cols = [
            'EmployeeId', 'SiteId', 'TaskId', 'EmployerId', 'CreatedBy', 
            'ScheduleWeek', 'Week', 'FKshiftId',
            'assignment_success', 'employee_satisfaction', 
            'employer_satisfaction', 'completion_likelihood',
            'processed_at'
        ]
        
        # Also exclude the original categorical columns (keep encoded versions)
        exclude_cols.extend(['WeekDay', 'Season', 'Month', 'Quarter'])
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_columns = feature_cols
        
        # Handle any remaining missing values
        df[feature_cols] = df[feature_cols].fillna(0)
        
        # Add feature metadata
        df['feature_engineering_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        logger.info(f"✅ Feature engineering completed!")
        logger.info(f"   📊 Total features: {len(feature_cols)}")
        logger.info(f"   🎯 Target variables: 4 (success, satisfaction, quality, completion)")
        logger.info(f"   📋 Feature categories:")
        
        # Count features by category
        temporal_features = [col for col in feature_cols if any(x in col for x in ['Day', 'Season', 'Month', 'Quarter', 'Holiday'])]
        employee_features = [col for col in feature_cols if col.startswith('emp_')]
        site_features = [col for col in feature_cols if col.startswith('site_')]
        task_features = [col for col in feature_cols if col.startswith('task_')]
        interaction_features = [col for col in feature_cols if any(x in col for x in ['experience', 'familiarity', 'workload'])]
        embedding_features = [col for col in feature_cols if 'embed' in col]
        
        logger.info(f"      📅 Temporal: {len(temporal_features)}")
        logger.info(f"      👥 Employee: {len(employee_features)}")
        logger.info(f"      📍 Site: {len(site_features)}")
        logger.info(f"      🛠️ Task: {len(task_features)}")
        logger.info(f"      🔗 Interaction: {len(interaction_features)}")
        logger.info(f"      🧠 Embeddings: {len(embedding_features)}")
        
        return df
    
    def save_engineered_features(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        Save the feature-engineered dataset
        """
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"feature_engineered_data_{timestamp}.csv"
        
        output_path = self.features_data_path / filename
        
        logger.info(f"💾 Saving feature-engineered data to: {output_path}")
        
        try:
            # Save main dataset
            df.to_csv(output_path, index=False)
            
            # Save latest version for easy access
            latest_path = self.features_data_path / "latest_features.csv"
            df.to_csv(latest_path, index=False)
            
            # Save feature metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'total_features': len(self.feature_columns),
                'total_records': len(df),
                'feature_columns': self.feature_columns,
                'target_columns': ['assignment_success', 'employee_satisfaction', 'employer_satisfaction', 'completion_likelihood'],
                'categorical_encoders': {k: list(v.classes_) for k, v in self.encoders.items()},
                'data_stats': {
                    'unique_employees': int(df['EmployeeId'].nunique()),
                    'unique_sites': int(df['SiteId'].nunique()),
                    'unique_tasks': int(df['TaskId'].nunique()),
                    'date_range': f"{df['ScheduleWeek'].min()} to {df['ScheduleWeek'].max()}"
                }
            }
            
            metadata_path = self.features_data_path / "feature_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Save embeddings to models folder (they're needed for training and prediction)
            models_path = Path("D:/Employee_Suggestions/employee_suggestion_system/models")
            models_path.mkdir(parents=True, exist_ok=True)
            embeddings_path = models_path / "embeddings.pkl"
            import joblib
            joblib.dump(self.employee_embeddings, embeddings_path)
            
            logger.info(f"✅ Features saved successfully!")
            logger.info(f"   📊 Records: {len(df):,}")
            logger.info(f"   🎯 Features: {len(self.feature_columns)}")
            logger.info(f"   📁 Files created:")
            logger.info(f"      - {output_path}")
            logger.info(f"      - {latest_path}")
            logger.info(f"      - {metadata_path}")
            logger.info(f"      - {embeddings_path}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to save features: {str(e)}")
            raise
    
    def run_feature_engineering(self, input_filename: str = None) -> Dict[str, Any]:
        """
        Complete feature engineering pipeline
        """
        
        logger.info("🚀 Starting feature engineering pipeline...")
        start_time = datetime.now()
        
        try:
            # Step 1: Load processed data
            df = self.load_processed_data(input_filename)
            
            # Step 2: Create target variables
            df = self.create_target_variables(df)
            
            # Step 3: Engineer temporal features
            df = self.engineer_temporal_features(df)
            
            # Step 4: Engineer employee-level features
            df = self.engineer_employee_features(df)
            
            # Step 5: Engineer site-level features
            df = self.engineer_site_features(df)
            
            # Step 6: Engineer task-level features  
            df = self.engineer_task_features(df)
            
            # Step 7: Engineer interaction features
            df = self.engineer_interaction_features(df)
            
            # Step 8: Engineer workload features
            df = self.engineer_workload_features(df)
            
            # Step 9: Create embeddings
            df = self.create_embeddings(df)
            
            # Step 10: Encode categorical features
            df = self.encode_categorical_features(df)
            
            # Step 11: Finalize features
            df = self.finalize_features(df)
            
            # Step 12: Save engineered features
            output_path = self.save_engineered_features(df)
            
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            
            result = {
                'status': 'success',
                'input_file': input_filename or 'latest_processed_data.csv',
                'output_file': output_path,
                'records_processed': len(df),
                'features_created': len(self.feature_columns),
                'duration_seconds': round(duration, 2),
                'summary': {
                    'total_records': len(df),
                    'total_features': len(self.feature_columns),
                    'unique_employees': df['EmployeeId'].nunique(),
                    'unique_sites': df['SiteId'].nunique(),
                    'unique_tasks': df['TaskId'].nunique(),
                    'date_range': f"{df['ScheduleWeek'].min()} to {df['ScheduleWeek'].max()}",
                    'target_stats': {
                        'avg_success_rate': round(df['assignment_success'].mean(), 3),
                        'avg_employee_satisfaction': round(df['employee_satisfaction'].mean(), 1),
                        'avg_employer_satisfaction': round(df['employer_satisfaction'].mean(), 1),
                        'avg_completion_likelihood': round(df['completion_likelihood'].mean(), 3)
                    }
                }
            }
            
            logger.info("🎉 Feature engineering completed successfully!")
            logger.info(f"   ⏱️  Duration: {duration:.2f} seconds")
            logger.info(f"   📊 Records: {len(df):,}")
            logger.info(f"   🎯 Features: {len(self.feature_columns)}")
            logger.info(f"   📁 Output: {output_path}")
            
            return result
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            
            logger.error(f"❌ Feature engineering failed after {duration:.2f} seconds")
            logger.error(f"   Error: {str(e)}")
            
            return {
                'status': 'failed',
                'error': str(e),
                'duration_seconds': round(duration, 2)
            }

def main():
    """Main function for running feature engineering"""
    
    print("🔧 Employee Schedule Feature Engineering")
    print("=" * 50)
    
    # Check for command line arguments
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h']:
            print("Usage:")
            print("  python feature_engineering.py                    # Use latest processed data")
            print("  python feature_engineering.py input_file.csv     # Specify input file")
            return
        
        input_file = sys.argv[1]
    else:
        input_file = None  # Use latest file
    
    # Initialize feature engineering pipeline
    feature_eng = FeatureEngineering()
    
    # Run feature engineering
    result = feature_eng.run_feature_engineering(input_file)
    
    # Print results
    if result['status'] == 'success':
        print(f"\n✅ SUCCESS!")
        print(f"📊 Processed {result['records_processed']:,} records")
        print(f"🎯 Created {result['features_created']} features")
        print(f"👥 {result['summary']['unique_employees']:,} unique employees")
        print(f"📍 {result['summary']['unique_sites']:,} unique sites")
        print(f"🛠️ {result['summary']['unique_tasks']:,} unique tasks")
        print(f"📅 Date range: {result['summary']['date_range']}")
        print(f"\n📈 Target Variable Stats:")
        print(f"   Success Rate: {result['summary']['target_stats']['avg_success_rate']:.1%}")
        print(f"   Employee Satisfaction: {result['summary']['target_stats']['avg_employee_satisfaction']:.1f}/10")
        print(f"   Employer Satisfaction: {result['summary']['target_stats']['avg_employer_satisfaction']:.1f}/10")
        print(f"   Completion Likelihood: {result['summary']['target_stats']['avg_completion_likelihood']:.1%}")
        print(f"\n💾 Saved to: {result['output_file']}")
        print(f"⏱️ Duration: {result['duration_seconds']:.2f} seconds")
    else:
        print(f"\n❌ FAILED!")
        print(f"Error: {result['error']}")
    
    return result

if __name__ == "__main__":
    result = main()