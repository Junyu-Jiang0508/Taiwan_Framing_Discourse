"""
Targeted sampling for remaining gaps after merge
Focus on hard-to-find labels: L1-10/11/12/13/14/15 and scarce L2
"""
import pandas as pd
import json
from pathlib import Path
from typing import Dict, List
import random

class TargetedGapSampler:
    def __init__(self, statistics_path: str):
        """Load merged statistics to identify gaps"""
        with open(statistics_path, 'r', encoding='utf-8') as f:
            self.stats = json.load(f)
        
        self.l1_gaps = self.stats['l1_gaps']
        self.l2_gaps = self.stats['l2_gaps']
    
    def load_refined_datasets(self) -> pd.DataFrame:
        """Load all refined datasets"""
        refined_dir = Path('01_Data/04_refined_datasets')
        
        all_data = []
        
        for dataset in ['01_news_datasets', '02_conference_datasets', '03_X_datasets']:
            dataset_dir = refined_dir / dataset
            if not dataset_dir.exists():
                continue
            
            for csv_file in dataset_dir.glob('*_units.csv'):
                if '_CONTEXT' in csv_file.name or '_DROPPED' in csv_file.name:
                    continue
                
                try:
                    df = pd.read_csv(csv_file, encoding='utf-8-sig')
                    df['source_file'] = csv_file.stem
                    all_data.append(df)
                except Exception as e:
                    pass
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
    
    def sample_for_priority_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Focus on hardest gaps:
        - L1-10/11/12: May not exist in codebook, skip
        - L1-13/14/15: Very scarce
        - L1-01/02/04/05/06/07: Medium gaps
        - Prioritize samples with scarce L2 keywords
        """
        
        # Enhanced keywords for hard-to-find labels
        priority_keywords = {
            'L1-13': ['競選', '竞选', '選舉', '选举', '選戰', '选战', '拉票', '催票', '文宣'],
            'L1-14': ['能力', '領導', '领导', '經驗', '经验', '執政', '执政', '團隊', '团队', '資格', '资格', '適任', '适任'],
            'L1-15': ['風險', '风险', '危機', '危机', '威脅', '威胁', '挑戰', '挑战', '不確定', '不确定', '擔憂', '担忧'],
            'L1-01': ['GDP', '經濟', '经济', '成長', '成长', '發展', '发展', '投資', '投资', '產業', '产业', '就業', '就业', '薪資', '薪资', '收入'],
            'L1-02': ['誠信', '诚信', '品德', '道德', '倫理', '伦理', '操守', '廉潔', '廉洁', '貪污', '贪污'],
            'L1-04': ['對立', '对立', '矛盾', '衝突', '冲突', '爭議', '争议', '對抗', '对抗', '分裂', '撕裂'],
            'L1-05': ['關懷', '关怀', '同情', '溫暖', '温暖', '照顧', '照顾', '弱勢', '弱势', '人權', '人权'],
            'L1-06': ['公平', '正義', '正义', '平等', '不公', '歧視', '歧视', '弱勢', '弱势', '剝削', '剥削'],
            'L1-07': ['合法', '違法', '违法', '法治', '法規', '法规', '憲法', '宪法', '司法', '依法'],
        }
        
        # Scarce L2 boost keywords
        scarce_l2_keywords = {
            'L2-03': ['繁榮', '繁荣', '富裕', '經濟', '经济', '產業', '产业', '競爭力', '竞争力'],
            'L2-05': ['公平', '正義', '正义', '福利', '社會', '社会', '弱勢', '弱势'],
            'L2-06': ['文化', '傳統', '传统', '歷史', '历史', '價值', '价值', '認同', '认同'],
            'L2-10': ['外交', '國際', '国际', '地位', '空間', '空间', '關係', '关系'],
            'L2-11': ['環保', '环保', '永續', '永续', '氣候', '气候', '綠能', '绿能', '生態', '生态'],
        }
        
        sampled = []
        
        # Priority order: hardest labels first
        priority_labels = ['L1-13', 'L1-14', 'L1-15', 'L1-01', 'L1-02', 'L1-04', 'L1-05', 'L1-06', 'L1-07']
        
        for label in priority_labels:
            if label not in self.l1_gaps:
                continue
            
            quota = self.l1_gaps[label]
            print(f"\nSampling for {label}: need {quota} samples")
            
            # Keyword matching
            keywords = priority_keywords.get(label, [])
            if not keywords:
                print(f"  No keywords defined for {label}, skipping")
                continue
            
            mask = df['sentence'].str.contains('|'.join(keywords), case=False, na=False)
            candidates = df[mask].copy()
            
            print(f"  Found {len(candidates)} keyword matches")
            
            # Add random samples if not enough
            if len(candidates) < quota * 2:
                additional_needed = quota * 2 - len(candidates)
                remaining = df[~mask]
                if len(remaining) > 0:
                    additional = remaining.sample(n=min(additional_needed, len(remaining)), random_state=42)
                    candidates = pd.concat([candidates, additional], ignore_index=True)
                    print(f"  Added {len(additional)} random samples")
            
            # Score with scarce L2 bonus
            def score_sample(row):
                score = 5.0  # Base score
                text = str(row.get('sentence', ''))
                
                # Scarce L2 bonus
                for l2_label, l2_kws in scarce_l2_keywords.items():
                    if l2_label in self.l2_gaps and any(kw in text for kw in l2_kws):
                        score += 2.0
                
                # Metadata bonuses
                if row.get('speakers'): score += 1.0
                if row.get('targets'): score += 0.8
                if row.get('role') == 'quote': score += 1.5
                elif row.get('role') == 'claim': score += 1.0
                
                # Length check
                if 50 <= len(text) <= 200: score += 0.5
                
                return score
            
            candidates['sample_score'] = candidates.apply(score_sample, axis=1)
            candidates = candidates.sort_values('sample_score', ascending=False)
            
            # Take top samples with diversity
            target_count = int(quota * 1.2)  # 20% oversampling
            selected = candidates.head(target_count).copy()
            
            selected['target_l1'] = label
            selected['sampling_reason'] = 'targeted_gap_filling'
            sampled.append(selected)
            
            print(f"  Selected {len(selected)} candidates ({len(selected)/quota:.1f}x quota)")
        
        if sampled:
            return pd.concat(sampled, ignore_index=True)
        return pd.DataFrame()
    
    def export_candidates(self, candidates: pd.DataFrame, output_dir: str):
        """Export sampling candidates"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Main file
        main_output = output_path / 'candidates_round2.csv'
        
        output_cols = [
            'unit_id', 'sentence', 'role', 'prev', 'next',
            'speakers', 'targets', 'source', 'date',
            'target_l1', 'sample_score', 'sampling_reason'
        ]
        
        available_cols = [col for col in output_cols if col in candidates.columns]
        candidates[available_cols].to_csv(main_output, index=False, encoding='utf-8-sig')
        
        print(f"\n[OK] Candidates saved: {main_output}")
        
        # Split by label
        for label in candidates['target_l1'].unique():
            label_df = candidates[candidates['target_l1'] == label]
            label_output = output_path / f'candidates_{label}_round2.csv'
            label_df[available_cols].to_csv(label_output, index=False, encoding='utf-8-sig')
            print(f"  [OK] {label}: {len(label_df)} candidates")

def main():
    stats_path = '01_Data/06_manual_sets/03_results/05_merged_statistics.json'
    
    print("=" * 80)
    print("TARGETED SAMPLING FOR REMAINING GAPS")
    print("=" * 80)
    
    sampler = TargetedGapSampler(stats_path)
    
    print(f"\nPriority L1 gaps:")
    priority_gaps = {k: v for k, v in sampler.l1_gaps.items() 
                     if k not in ['L1-10', 'L1-11', 'L1-12']}  # Skip non-existent labels
    
    for label, gap in sorted(priority_gaps.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {label}: +{gap}")
    
    print(f"\nPriority L2 gaps:")
    for label, gap in sorted(sampler.l2_gaps.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {label}: +{gap}")
    
    print("\n" + "=" * 80)
    print("LOADING DATASETS")
    print("=" * 80)
    
    df = sampler.load_refined_datasets()
    print(f"Available samples: {len(df)}")
    
    print("\n" + "=" * 80)
    print("SAMPLING CANDIDATES")
    print("=" * 80)
    
    candidates = sampler.sample_for_priority_gaps(df)
    
    if len(candidates) == 0:
        print("No candidates sampled")
        return
    
    print(f"\nTotal candidates: {len(candidates)}")
    print(f"Average score: {candidates['sample_score'].mean():.2f}")
    
    # Export
    output_dir = '01_Data/06_manual_sets/02_sampling_candidates'
    sampler.export_candidates(candidates, output_dir)
    
    print("\n" + "=" * 80)
    print("SAMPLING COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Prepare candidates for annotation:")
    print("   python 08_prepare_candidates_for_annotation.py (update for round2)")
    print("2. Run annotation:")
    print("   python 05_validation_deepseek_annotation_async.py (update path)")
    print("3. Merge with existing 792 samples")
    print("4. Check if gaps are filled")

if __name__ == "__main__":
    main()

