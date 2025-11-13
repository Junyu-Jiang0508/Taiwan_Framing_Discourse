"""
Create validation set from refined datasets (post-filtering)
Input: 01_Data/04_refined_datasets/ (after REVIEW post-filtering)
Output: 01_Data/06_manual_sets/01_datasets/02_validation_set.csv
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

class ValidationSetCreator:
    def __init__(self, election_date='2024-01-13'):
        self.election_date = pd.to_datetime(election_date)
        self.output_dir = Path('01_Data/06_manual_sets/01_datasets')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_news_data(self):
        """Load news data from refined datasets (after post-filtering)"""
        news_dir = Path('01_Data/04_refined_datasets/01_news_datasets')
        all_news = []
        
        # Only load main refined files (not CONTEXT or DROPPED)
        for csv_file in news_dir.glob('*_units.csv'):
            if '_CONTEXT' in csv_file.name or '_DROPPED' in csv_file.name:
                continue
                
            try:
                df = pd.read_csv(csv_file, encoding='utf-8-sig')
                df['source_type'] = 'news'
                
                if '01_Ho' in csv_file.name:
                    df['party'] = 'KMT'
                    df['candidate'] = '侯友宜'
                elif '02_Ke' in csv_file.name:
                    df['party'] = 'TPP'
                    df['candidate'] = '柯文哲'
                elif '03_Lai' in csv_file.name:
                    df['party'] = 'DPP'
                    df['candidate'] = '赖清德'
                else:
                    continue
                
                all_news.append(df)
            except Exception as e:
                print(f"Error loading {csv_file.name}: {e}")
        
        if all_news:
            return pd.concat(all_news, ignore_index=True)
        return pd.DataFrame()
    
    def load_x_data(self):
        """Load X (Twitter) data from refined datasets (after post-filtering)"""
        x_dir = Path('01_Data/04_refined_datasets/03_X_datasets')
        all_x = []
        
        # Only load main refined files (not CONTEXT or DROPPED)
        for csv_file in x_dir.glob('*_units.csv'):
            if '_CONTEXT' in csv_file.name or '_DROPPED' in csv_file.name:
                continue
                
            try:
                df = pd.read_csv(csv_file, encoding='utf-8-sig')
                df['source_type'] = 'social_media'
                
                if '01_Ho' in csv_file.name or '01_Hou' in csv_file.name:
                    df['party'] = 'KMT'
                    df['candidate'] = '侯友宜'
                elif '02_Ke' in csv_file.name:
                    df['party'] = 'TPP'
                    df['candidate'] = '柯文哲'
                elif '03_Lai' in csv_file.name:
                    df['party'] = 'DPP'
                    df['candidate'] = '赖清德'
                else:
                    continue
                
                if 'Tweet Date' in df.columns:
                    df['date'] = pd.to_datetime(df['Tweet Date'], errors='coerce')
                
                if 'Tweet Content' in df.columns and 'sentence' not in df.columns:
                    df['sentence'] = df['Tweet Content']
                if 'Tweet Content' in df.columns and 'content' not in df.columns:
                    df['content'] = df['Tweet Content']
                
                all_x.append(df)
            except Exception as e:
                print(f"Error loading {csv_file.name}: {e}")
        
        if all_x:
            return pd.concat(all_x, ignore_index=True)
        return pd.DataFrame()
    
    def load_conference_data(self):
        """Load conference/debate data from refined datasets (after post-filtering)"""
        conf_dir = Path('01_Data/04_refined_datasets/02_conference_datasets')
        all_conf = []
        
        # Only load main refined files (not CONTEXT or DROPPED)
        for csv_file in conf_dir.glob('*_units.csv'):
            if '_CONTEXT' in csv_file.name or '_DROPPED' in csv_file.name:
                continue
                
            try:
                df = pd.read_csv(csv_file, encoding='utf-8-sig')
                df['source_type'] = 'conference'
                df['date'] = self.election_date - timedelta(days=30)
                
                if '01_Ho' in csv_file.name:
                    df['party'] = 'KMT'
                    df['candidate'] = '侯友宜'
                elif '02_Ke' in csv_file.name:
                    df['party'] = 'TPP'
                    df['candidate'] = '柯文哲'
                elif '03_Lai' in csv_file.name:
                    df['party'] = 'DPP'
                    df['candidate'] = '赖清德'
                else:
                    continue
                
                all_conf.append(df)
            except Exception as e:
                print(f"Error loading {csv_file.name}: {e}")
        
        if all_conf:
            return pd.concat(all_conf, ignore_index=True)
        return pd.DataFrame()
    
    def classify_time_period(self, date):
        if pd.isna(date):
            return 'unknown'
        
        try:
            date = pd.to_datetime(date)
            days_before = (self.election_date - date).days
            
            if days_before < 90:
                return 'within_3m'
            elif days_before < 180:
                return 'within_6m'
            else:
                return 'over_6m'
        except:
            return 'unknown'
    
    def stratified_sample(self, df, candidate, n_samples=100):
        candidate_df = df[df['candidate'] == candidate].copy()
        
        if len(candidate_df) == 0:
            print(f"No data for {candidate}")
            return pd.DataFrame()
        
        source_quotas = {
            'news': int(n_samples * 0.5),
            'conference': int(n_samples * 0.3),
            'social_media': int(n_samples * 0.1)
        }
        
        remaining = n_samples - sum(source_quotas.values())
        sampled_data = []
        
        for source_type, quota in source_quotas.items():
            source_df = candidate_df[candidate_df['source_type'] == source_type]
            
            if len(source_df) == 0:
                print(f"  No {source_type} data for {candidate}, redistributing quota")
                remaining += quota
                continue
            
            if source_type == 'conference':
                time_sample = source_df.sample(n=min(quota, len(source_df)), random_state=42)
            else:
                source_df['time_period'] = source_df['date'].apply(self.classify_time_period)
                time_periods = ['within_3m', 'within_6m', 'over_6m']
                time_quota = quota // 3
                time_sample = pd.DataFrame()
                
                for period in time_periods:
                    period_df = source_df[source_df['time_period'] == period]
                    if len(period_df) > 0:
                        n = min(time_quota, len(period_df))
                        sampled = period_df.sample(n=n, random_state=42)
                        time_sample = pd.concat([time_sample, sampled])
                
                time_remaining = quota - len(time_sample)
                if time_remaining > 0:
                    unused_df = source_df[~source_df.index.isin(time_sample.index)]
                    if len(unused_df) > 0:
                        extra = unused_df.sample(n=min(time_remaining, len(unused_df)), random_state=42)
                        time_sample = pd.concat([time_sample, extra])
            
            sampled_data.append(time_sample)
        
        if remaining > 0:
            used_indices = pd.concat(sampled_data).index if sampled_data else []
            remaining_df = candidate_df[~candidate_df.index.isin(used_indices)]
            if len(remaining_df) > 0:
                extra_sample = remaining_df.sample(n=min(remaining, len(remaining_df)), random_state=42)
                sampled_data.append(extra_sample)
        
        if sampled_data:
            result = pd.concat(sampled_data, ignore_index=True)
            return result.head(n_samples)
        
        return pd.DataFrame()
    
    def create_validation_set(self):
        print("Loading data sources...")
        print("=" * 50)
        
        news_df = self.load_news_data()
        print(f"Loaded {len(news_df)} news records")
        
        x_df = self.load_x_data()
        print(f"Loaded {len(x_df)} social media records")
        
        conf_df = self.load_conference_data()
        print(f"Loaded {len(conf_df)} conference records")
        
        all_data = pd.concat([news_df, x_df, conf_df], ignore_index=True)
        all_data['date'] = pd.to_datetime(all_data['date'], errors='coerce')
        
        print(f"\nTotal records available: {len(all_data)}")
        
        print("\n" + "=" * 50)
        print("Stratified sampling...")
        
        candidates = ['侯友宜', '柯文哲', '赖清德']
        all_samples = []
        
        for candidate in candidates:
            print(f"\nSampling for {candidate} (target: 100 samples)...")
            sample = self.stratified_sample(all_data, candidate, n_samples=100)
            all_samples.append(sample)
            print(f"  Actual sampled: {len(sample)} records")
            if len(sample) > 0:
                print(f"  By source: {dict(sample['source_type'].value_counts())}")
        
        validation_set = pd.concat(all_samples, ignore_index=True)
        
        remaining = 400 - len(validation_set)
        if remaining > 0:
            print(f"\nAdding {remaining} samples to reach 400...")
            for candidate in candidates:
                if remaining <= 0:
                    break
                add_n = remaining // len(candidates)
                candidate_df = all_data[all_data['candidate'] == candidate]
                used = validation_set[validation_set['candidate'] == candidate]
                unused = candidate_df[~candidate_df.index.isin(used.index)]
                if len(unused) > 0:
                    extra = unused.sample(n=min(add_n, len(unused)), random_state=42)
                    validation_set = pd.concat([validation_set, extra], ignore_index=True)
                    remaining -= len(extra)
        
        validation_set = validation_set.head(400).copy()
        
        output_cols = ['candidate', 'party', 'source_type', 'date', 'sentence', 'content']
        available_cols = [col for col in output_cols if col in validation_set.columns]
        validation_set = validation_set[available_cols]
        
        validation_set.insert(0, 'id', range(1, len(validation_set) + 1))
        
        output_path = self.output_dir / '02_validation_set.csv'
        validation_set.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print("\n" + "=" * 50)
        print("=== Final Validation Set Summary ===")
        print(f"Total samples: {len(validation_set)}")
        print(f"\nBy candidate:")
        print(validation_set['candidate'].value_counts())
        print(f"\nBy source type:")
        print(validation_set['source_type'].value_counts())
        
        validation_set['time_period'] = validation_set['date'].apply(self.classify_time_period)
        print(f"\nBy time period:")
        print(validation_set['time_period'].value_counts())
        
        print(f"\nSaved to: {output_path}")
        
        return validation_set


def main():
    creator = ValidationSetCreator(election_date='2024-01-13')
    validation_set = creator.create_validation_set()


if __name__ == "__main__":
    main()

