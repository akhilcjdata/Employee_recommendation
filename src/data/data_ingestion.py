"""
Simple Data Ingestion Pipeline
Loads CSV data from data/raw folder for ML recommendation system
"""

import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataIngestion:
    """Simple data ingestion for shift recommendation system"""
    
    def __init__(self, raw_data_path: str = "D:/Employee_Suggestions/employee_suggestion_system/data/raw", processed_data_path: str = "D:/Employee_Suggestions/employee_suggestion_system/data/processed"):
        self.raw_data_path = Path(raw_data_path)
        self.processed_data_path = Path(processed_data_path)
        
        # Create directories if they don't exist
        self.processed_data_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Raw data path: {self.raw_data_path}")
        logger.info(f"Processed data path: {self.processed_data_path}")
    
    def load_raw_data(self, filename: str = "EmployeeScheduleData 2.csv") -> pd.DataFrame:
        """
        Load raw CSV data from data/raw folder
        
        Args:
            filename: Name of the CSV file to load
            
        Returns:
            pandas.DataFrame: Loaded raw data
        """
        
        file_path = self.raw_data_path / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        logger.info(f"Loading data from: {file_path}")
        
        try:
            # Load the CSV file
            df = pd.read_csv(file_path)
            
            logger.info(f"✅ Data loaded successfully!")
            logger.info(f"   📊 Shape: {df.shape}")
            logger.info(f"   📋 Columns: {list(df.columns)}")
            logger.info(f"   📅 Date range: {df['ScheduleWeek'].min()} to {df['ScheduleWeek'].max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to load data: {str(e)}")
            raise
    
    def basic_data_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Basic data cleaning and preparation
        
        Args:
            df: Raw dataframe
            
        Returns:
            pandas.DataFrame: Cleaned dataframe
        """
        
        logger.info("🧹 Starting basic data cleaning...")
        
        # Make a copy
        df_clean = df.copy()
        
        # Basic info about the data
        logger.info(f"   Input records: {len(df_clean):,}")
        logger.info(f"   Missing values: {df_clean.isnull().sum().sum()}")
        logger.info(f"   Duplicate rows: {df_clean.duplicated().sum()}")
        
        # Remove any completely null rows
        df_clean = df_clean.dropna(how='all')
        
        # Remove duplicates
        before_dedup = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        after_dedup = len(df_clean)
        
        if before_dedup != after_dedup:
            logger.info(f"   Removed {before_dedup - after_dedup:,} duplicate records")
        
        # Fix ScheduleWeek format if needed
        if 'ScheduleWeek' in df_clean.columns:
            # Ensure it's properly formatted as 6-digit number (YYYYWW)
            df_clean['ScheduleWeek'] = df_clean['ScheduleWeek'].astype(str).str.zfill(6)
            
            # Extract year and week number
            df_clean['Schedule_Year'] = df_clean['ScheduleWeek'].str[:4].astype(int)
            df_clean['Schedule_WeekNumber'] = df_clean['ScheduleWeek'].str[4:].astype(int)
        
        # Add some derived columns that might be useful
        df_clean['IsWeekend'] = df_clean['WeekDay'].isin([6, 7]).astype(int)
        
        # Add processing timestamp
        df_clean['processed_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        logger.info(f"✅ Data cleaning completed!")
        logger.info(f"   Output records: {len(df_clean):,}")
        logger.info(f"   Final columns: {len(df_clean.columns)}")
        
        return df_clean
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = None) -> str:
        """
        Save processed data to processed folder
        
        Args:
            df: Cleaned dataframe
            filename: Output filename (optional)
            
        Returns:
            str: Path to saved file
        """
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"processed_employee_schedule_data_{timestamp}.csv"
        
        output_path = self.processed_data_path / filename
        
        logger.info(f"💾 Saving processed data to: {output_path}")
        
        try:
            # Save as CSV
            df.to_csv(output_path, index=False)
            
            # Also save as 'latest' for easy access
            latest_path = self.processed_data_path / "latest_processed_data.csv"
            df.to_csv(latest_path, index=False)
            
            # Save basic statistics
            stats = {
                'timestamp': datetime.now().isoformat(),
                'total_records': len(df),
                'columns': list(df.columns),
                'date_range': {
                    'min_schedule_week': int(df['ScheduleWeek'].min()),
                    'max_schedule_week': int(df['ScheduleWeek'].max())
                },
                'unique_counts': {
                    'employees': int(df['EmployeeId'].nunique()),
                    'employers': int(df['EmployerId'].nunique()),
                    'sites': int(df['SiteId'].nunique()),
                    'tasks': int(df['TaskId'].nunique()),
                    'shifts': int(df['FKshiftId'].nunique())
                }
            }
            
            stats_path = self.processed_data_path / "data_statistics.json"
            import json
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"✅ Data saved successfully!")
            logger.info(f"   📊 Records: {len(df):,}")
            logger.info(f"   👥 Unique employees: {stats['unique_counts']['employees']:,}")
            logger.info(f"   🏢 Unique employers: {stats['unique_counts']['employers']:,}")
            logger.info(f"   📍 Unique sites: {stats['unique_counts']['sites']:,}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to save data: {str(e)}")
            raise
    
    def run_ingestion(self, input_filename: str = "EmployeeScheduleData 2.csv") -> Dict[str, Any]:
        """
        Complete data ingestion pipeline
        
        Args:
            input_filename: Name of input CSV file
            
        Returns:
            dict: Results of ingestion process
        """
        
        logger.info("🚀 Starting data ingestion pipeline...")
        start_time = datetime.now()
        
        try:
            # Step 1: Load raw data
            raw_data = self.load_raw_data(input_filename)
            
            # Step 2: Clean data
            processed_data = self.basic_data_cleaning(raw_data)
            
            # Step 3: Save processed data
            output_path = self.save_processed_data(processed_data)
            
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            
            result = {
                'status': 'success',
                'input_file': input_filename,
                'output_file': output_path,
                'records_processed': len(processed_data),
                'duration_seconds': round(duration, 2),
                'summary': {
                    'total_records': len(processed_data),
                    'unique_employees': processed_data['EmployeeId'].nunique(),
                    'unique_employers': processed_data['EmployerId'].nunique(),
                    'unique_sites': processed_data['SiteId'].nunique(),
                    'date_range': f"{processed_data['ScheduleWeek'].min()} to {processed_data['ScheduleWeek'].max()}"
                }
            }
            
            logger.info("🎉 Data ingestion completed successfully!")
            logger.info(f"   ⏱️  Duration: {duration:.2f} seconds")
            logger.info(f"   📊 Records: {len(processed_data):,}")
            logger.info(f"   📁 Output: {output_path}")
            
            return result
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            
            logger.error(f"❌ Data ingestion failed after {duration:.2f} seconds")
            logger.error(f"   Error: {str(e)}")
            
            return {
                'status': 'failed',
                'error': str(e),
                'duration_seconds': round(duration, 2)
            }

def main():
    """Main function for running data ingestion"""
    
    print("🔄 Employee Schedule Data Ingestion")
    print("=" * 50)
    
    # Check if command line arguments are provided
    import sys
    
    if len(sys.argv) > 1:
        # Handle command line arguments
        if sys.argv[1] in ['--help', '-h']:
            print("Usage:")
            print("  python data_ingestion.py                    # Use default paths")
            print("  python data_ingestion.py input_file         # Specify input file")
            print("  python data_ingestion.py input_file output  # Specify both paths")
            return
        
        input_file = sys.argv[1] if len(sys.argv) > 1 else "EmployeeScheduleData 2.csv"
        
        # Initialize ingestion pipeline
        if len(sys.argv) > 2:
            # Custom output path provided
            output_dir = sys.argv[2]
            ingestion = DataIngestion(processed_data_path=output_dir)
        else:
            ingestion = DataIngestion()
        
        # Run the ingestion with specified input file
        result = ingestion.run_ingestion(input_file)
    else:
        # Default behavior - no command line args
        ingestion = DataIngestion()
        result = ingestion.run_ingestion()
    
    # Print results
    if result['status'] == 'success':
        print(f"\n✅ SUCCESS!")
        print(f"📊 Processed {result['records_processed']:,} records")
        print(f"👥 {result['summary']['unique_employees']:,} unique employees")
        print(f"🏢 {result['summary']['unique_employers']:,} unique employers")
        print(f"📍 {result['summary']['unique_sites']:,} unique sites")
        print(f"📅 Date range: {result['summary']['date_range']}")
        print(f"💾 Saved to: {result['output_file']}")
    else:
        print(f"\n❌ FAILED!")
        print(f"Error: {result['error']}")
    
    return result

if __name__ == "__main__":
    result = main()