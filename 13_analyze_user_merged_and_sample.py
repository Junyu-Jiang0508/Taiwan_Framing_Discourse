"""
Analyze user's merged annotations (05_merged_annotations.csv)
Identify remaining gaps and generate Round 3 sampling
"""
import pandas as pd
import json
from collections import Counter
from pathlib import Path

def analyze_merged(csv_path: str):
    """Analyze the user-merged annotation file"""
    
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    
    print("=" * 80)
    print("ANALYSIS OF USER-MERGED ANNOTATIONS")
    print("=" * 80)
    print(f"\nTotal samples: {len(df)}")
    
    # L1 distribution (only 9 valid labels)
    l1_counts = df['L1_label'].value_counts()
    l1_target = 100
    
    print(f"\n{'='*80}")
    print("L1 DISTRIBUTION (Target: 100 per label)")
    print(f"{'='*80}")
    print(f"\n{'Label':<15} {'Count':>10} {'Gap':>10} {'Status':<15}")
    print("-" * 80)
    
    l1_gaps = {}
    for i in range(1, 10):  # L1-01 to L1-09 only
        label = f'L1-{i:02d}'
        count = l1_counts.get(label, 0)
        gap = max(0, l1_target - count)
        
        if gap > 0:
            l1_gaps[label] = gap
            status = f'Need +{gap}'
        else:
            status = f'OK (+{count - l1_target})'
        
        print(f"{label:<15} {count:>10} {gap:>10} {status:<15}")
    
    total_l1_gap = sum(l1_gaps.values())
    print(f"\nTotal L1 gap: +{total_l1_gap} samples")
    
    # Check for invalid L1 labels
    invalid_l1 = []
    for label in l1_counts.index:
        if pd.notna(label):
            try:
                num = int(label.replace('L1-', ''))
                if num > 9:
                    invalid_l1.append((label, l1_counts[label]))
            except:
                pass
    
    if invalid_l1:
        print(f"\n[WARNING] Invalid L1 labels found:")
        for label, count in invalid_l1:
            print(f"  {label}: {count} samples (should not exist!)")
        print(f"  These need to be corrected in the prompt!")
    
    # L2 distribution (15 valid labels)
    all_l2 = []
    for l2_str in df['L2_labels'].dropna():
        if l2_str and l2_str != '':
            labels = [l.strip() for l in str(l2_str).split('|') if l.strip()]
            all_l2.extend(labels)
    
    l2_counts = Counter(all_l2)
    l2_target = 75
    
    print(f"\n{'='*80}")
    print("L2 DISTRIBUTION (Target: 75 per label)")
    print(f"{'='*80}")
    print(f"\nTotal L2 instances: {len(all_l2)}")
    print(f"Average per sample: {len(all_l2)/len(df):.2f}")
    
    print(f"\n{'Label':<15} {'Count':>10} {'Gap':>10} {'Status':<15}")
    print("-" * 80)
    
    l2_gaps = {}
    for i in range(1, 16):  # L2-01 to L2-15
        label = f'L2-{i:02d}'
        count = l2_counts.get(label, 0)
        gap = max(0, l2_target - count)
        
        if gap > 0:
            l2_gaps[label] = gap
            status = f'Need +{gap}'
        else:
            status = f'OK (+{count - l2_target})'
        
        print(f"{label:<15} {count:>10} {gap:>10} {status:<15}")
    
    total_l2_gap = sum(l2_gaps.values())
    avg_l2 = len(all_l2) / len(df) if len(df) > 0 else 1.8
    estimated_for_l2 = int(total_l2_gap / avg_l2) if avg_l2 > 0 else total_l2_gap
    
    print(f"\nTotal L2 gap: +{total_l2_gap} instances")
    print(f"Estimated samples needed for L2: ~{estimated_for_l2}")
    
    # Final recommendation
    print(f"\n{'='*80}")
    print("ROUND 3 SAMPLING RECOMMENDATION")
    print(f"{'='*80}")
    
    recommended_samples = max(total_l1_gap, estimated_for_l2)
    
    print(f"\nBased on gaps:")
    print(f"  L1 needs: +{total_l1_gap} samples")
    print(f"  L2 needs: ~{estimated_for_l2} samples")
    print(f"\nRecommended Round 3 sampling: +{recommended_samples} samples")
    
    # Priority labels for Round 3
    priority_l1 = sorted(l1_gaps.items(), key=lambda x: x[1], reverse=True)
    priority_l2 = sorted(l2_gaps.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nPriority L1 labels (top 5):")
    for label, gap in priority_l1[:5]:
        print(f"  {label}: +{gap}")
    
    print(f"\nPriority L2 labels (top 5):")
    for label, gap in priority_l2[:5]:
        print(f"  {label}: +{gap}")
    
    # Save analysis
    summary = {
        'total_samples': int(len(df)),
        'l1_distribution': {k: int(v) for k, v in l1_counts.items()},
        'l2_distribution': {k: int(v) for k, v in l2_counts.items()},
        'l2_avg_per_sample': float(avg_l2),
        'l1_gaps': {k: int(v) for k, v in l1_gaps.items()},
        'l2_gaps': {k: int(v) for k, v in l2_gaps.items()},
        'total_l1_gap': int(total_l1_gap),
        'total_l2_gap': int(total_l2_gap),
        'recommended_round3_samples': int(recommended_samples),
        'invalid_l1_labels': [{'label': label, 'count': int(count)} for label, count in invalid_l1]
    }
    
    output_path = csv_path.replace('.csv', '_gap_analysis.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n[OK] Gap analysis saved: {output_path}")
    
    return summary

def sample_for_l2_gaps(df: pd.DataFrame, l2_gaps: dict) -> pd.DataFrame:
    """
    Dedicated sampling for scarce L2 labels
    Focus on L2-03, L2-05, L2-06, L2-10
    """
    
    # Enhanced L2-specific keywords
    l2_specific_keywords = {
        'L2-03': ['繁榮', '繁荣', '富裕', '富強', '富强', '經濟成就', '经济成就', '興旺', '兴旺', '強盛', '强盛', '財富', '财富', '經濟實力', '经济实力', '經濟奇蹟', '经济奇迹'],
        'L2-05': ['歷史', '历史', '過去', '过去', '先人', '祖先', '傳承', '传承', '歷史意義', '历史意义', '歷史事件', '历史事件', '解讀', '解读', '詮釋', '诠释'],
        'L2-06': ['記憶', '记忆', '不忘', '銘記', '铭记', '紀念', '纪念', '回憶', '回忆', '共同經驗', '共同经验', '集體', '集体', '記得', '记得', '難忘', '难忘'],
        'L2-10': ['感動', '感动', '溫暖', '温暖', '振奮', '振奋', '鼓舞', '激勵', '激励', '感受', '心情', '情感', '熱情', '热情', '感懷', '感怀'],
    }
    
    sampled = []
    
    # Only sample for L2 gaps that are significant (>20)
    priority_l2 = {k: v for k, v in l2_gaps.items() if v > 20}
    
    print(f"\n{'='*80}")
    print("DEDICATED L2 GAP SAMPLING")
    print(f"{'='*80}")
    
    for label, gap in sorted(priority_l2.items(), key=lambda x: x[1], reverse=True):
        keywords = l2_specific_keywords.get(label, [])
        if not keywords:
            continue
        
        # Need more samples than gap because of avg 2 L2/sample
        quota = int(gap * 0.6)  # Sample 60% of gap (assuming multiple L2 per sample)
        
        print(f"\nSampling for {label}: gap={gap}, sampling {quota} candidates")
        
        mask = df['sentence'].str.contains('|'.join(keywords), case=False, na=False)
        candidates = df[mask].copy()
        
        print(f"  Found {len(candidates)} keyword matches")
        
        if len(candidates) == 0:
            continue
        
        # Score candidates
        def score_l2(row):
            score = 8.0  # Higher base for L2-specific
            text = str(row.get('sentence', ''))
            
            # Check for keyword density
            kw_count = sum(1 for kw in keywords if kw in text)
            score += kw_count * 1.5
            
            # Bonuses
            if row.get('speakers'): score += 1.0
            if row.get('targets'): score += 0.8
            if row.get('role') == 'quote': score += 2.0
            elif row.get('role') == 'claim': score += 1.5
            
            if 50 <= len(text) <= 200: score += 0.5
            
            return score
        
        candidates['sample_score'] = candidates.apply(score_l2, axis=1)
        candidates = candidates.sort_values('sample_score', ascending=False)
        
        # Select top candidates
        target_count = min(int(quota * 1.5), len(candidates))
        selected = candidates.head(target_count).copy()
        
        selected['target_l2'] = label
        selected['sampling_reason'] = 'l2_gap_filling'
        sampled.append(selected)
        
        print(f"  Selected {len(selected)} candidates")
    
    if sampled:
        return pd.concat(sampled, ignore_index=True)
    return pd.DataFrame()

def sample_for_round3(gaps_summary: dict):
    """Generate Round 3 sampling based on L1 AND L2 gaps"""
    
    # Load refined datasets
    refined_dir = Path('01_Data/04_refined_datasets')
    
    print(f"\n{'='*80}")
    print("LOADING REFINED DATASETS FOR ROUND 3")
    print(f"{'='*80}")
    
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
            except:
                pass
    
    if not all_data:
        print("Error: No data available")
        return None
    
    df = pd.concat(all_data, ignore_index=True)
    print(f"Available samples: {len(df)}")
    
    # Enhanced keywords based on actual L1 definitions
    l1_keywords = {
        'L1-01': ['經濟', '经济', 'GDP', '通膨', '失業', '失业', '財政', '财政', '貿易', '贸易', '投資', '投资', '產業', '产业', '市場', '市场', '價格', '价格', '供應鏈', '供应链'],
        'L1-02': ['道德', '倫理', '伦理', '價值觀', '价值观', '誠信', '诚信', '操守', '責任', '责任', '善惡', '善恶', '應該', '应该'],
        'L1-04': ['生活', '健康', '醫療', '医疗', '住房', '社保', '教育', '文化', '環保', '环保', '食品', '交通', '安全', '福祉', '公衛', '公卫'],
        'L1-05': ['公平', '平等', '正義', '正义', '歧視', '歧视', '弱勢', '弱势', '貧富', '贫富', '分配', '權利', '权利', '保障'],
        'L1-06': ['法律', '法治', '憲法', '宪法', '司法', '立法', '合法', '違法', '违法', '依法', '法規', '法规', '政府', '行政', '權力', '权力'],
        'L1-07': ['憤怒', '愤怒', '震驚', '震惊', '悲痛', '希望', '恐懼', '恐惧', '熱情', '热情', '同情', '激動', '激动', '情緒', '情绪', '感受'],
    }
    
    # Scarce L2 keywords
    l2_keywords = {
        'L2-03': ['繁榮', '繁荣', '富裕', '強盛', '强盛', '興旺', '兴旺', '經濟成就', '经济成就', '財富', '财富'],
        'L2-05': ['歷史', '历史', '過去', '过去', '先人', '祖先', '傳統', '传统', '延續', '延续', '歷史意義', '历史意义'],
        'L2-06': ['記憶', '记忆', '不忘', '銘記', '铭记', '紀念', '纪念', '回憶', '回忆', '共同經驗', '共同经验'],
        'L2-10': ['感動', '感动', '溫暖', '温暖', '振奮', '振奋', '鼓舞', '激勵', '激励', '感受', '心情'],
        'L2-11': ['驕傲', '骄傲', '自豪', '榮耀', '荣耀', '光榮', '光荣', '領先', '领先', '卓越', '成就', '優勢', '优势'],
    }
    
    print(f"\n{'='*80}")
    print("SAMPLING FOR ROUND 3 GAPS")
    print(f"{'='*80}")
    
    l1_gaps = gaps_summary['l1_gaps']
    l2_gaps = gaps_summary['l2_gaps']
    
    sampled = []
    
    # Sample for each L1 gap
    for label, quota in sorted(l1_gaps.items(), key=lambda x: x[1], reverse=True):
        print(f"\nSampling for {label}: need +{quota}")
        
        keywords = l1_keywords.get(label, [])
        if keywords:
            mask = df['sentence'].str.contains('|'.join(keywords), case=False, na=False)
            candidates = df[mask].copy()
        else:
            candidates = df.sample(n=min(quota * 3, len(df)), random_state=42)
        
        print(f"  Found {len(candidates)} keyword matches")
        
        # Add random if not enough
        if len(candidates) < quota * 2:
            additional_needed = quota * 2 - len(candidates)
            remaining = df[~df.index.isin(candidates.index)]
            if len(remaining) > 0:
                additional = remaining.sample(n=min(additional_needed, len(remaining)), random_state=42)
                candidates = pd.concat([candidates, additional])
        
        # Score with L2 gap consideration
        def score_sample(row):
            score = 5.0
            text = str(row.get('sentence', ''))
            
            # Bonus for scarce L2
            for l2_label, l2_kws in l2_keywords.items():
                if l2_label in l2_gaps and any(kw in text for kw in l2_kws):
                    score += 2.0
            
            # Metadata bonuses
            if row.get('speakers'): score += 1.0
            if row.get('targets'): score += 0.8
            if row.get('role') == 'quote': score += 1.5
            elif row.get('role') == 'claim': score += 1.0
            
            # Length
            if 50 <= len(text) <= 200: score += 0.5
            
            return score
        
        candidates['sample_score'] = candidates.apply(score_sample, axis=1)
        candidates = candidates.sort_values('sample_score', ascending=False)
        
        # Take top with 1.3x oversampling
        target_count = int(quota * 1.3)
        selected = candidates.head(target_count).copy()
        
        selected['target_l1'] = label
        selected['sampling_reason'] = 'round3_gap_filling'
        sampled.append(selected)
        
        print(f"  Selected {len(selected)} candidates")
    
    if sampled:
        result_df = pd.concat(sampled, ignore_index=True)
        
        # Export
        output_dir = Path('01_Data/06_manual_sets/02_sampling_candidates')
        output_path = output_dir / 'candidates_round3.csv'
        
        output_cols = [
            'unit_id', 'sentence', 'role', 'prev', 'next',
            'speakers', 'targets', 'source', 'date',
            'target_l1', 'sample_score', 'sampling_reason'
        ]
        
        available_cols = [col for col in output_cols if col in result_df.columns]
        result_df[available_cols].to_csv(output_path, index=False, encoding='utf-8-sig')
        
        print(f"\n[OK] Round 3 candidates saved: {output_path}")
        
        # Split by label
        for label in result_df['target_l1'].unique():
            label_df = result_df[result_df['target_l1'] == label]
            label_output = output_dir / f'candidates_{label}_round3.csv'
            label_df[available_cols].to_csv(label_output, index=False, encoding='utf-8-sig')
            print(f"  [OK] {label}: {len(label_df)} candidates")
        
        return result_df
    
    return None

def main():
    csv_path = '01_Data/06_manual_sets/03_results/05_merged_annotations.csv'
    
    try:
        # Analyze current state
        summary = analyze_merged(csv_path)
        
        # Generate Round 3 sampling if gaps exist
        if summary['total_l1_gap'] > 0 or summary['total_l2_gap'] > 0:
            # Sample for L1 gaps
            l1_candidates = sample_for_round3(summary)
            
            # Sample for L2 gaps (dedicated)
            from pathlib import Path
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
                        df_temp = pd.read_csv(csv_file, encoding='utf-8-sig')
                        df_temp['source_file'] = csv_file.stem
                        all_data.append(df_temp)
                    except:
                        pass
            
            if all_data:
                df_all = pd.concat(all_data, ignore_index=True)
                l2_candidates = sample_for_l2_gaps(df_all, summary['l2_gaps'])
            else:
                l2_candidates = None
            
            # Merge L1 and L2 candidates
            candidates_list = []
            if l1_candidates is not None and len(l1_candidates) > 0:
                candidates_list.append(l1_candidates)
            if l2_candidates is not None and len(l2_candidates) > 0:
                candidates_list.append(l2_candidates)
            
            if candidates_list:
                # Combine and deduplicate
                candidates = pd.concat(candidates_list, ignore_index=True)
                candidates = candidates.drop_duplicates(subset='unit_id', keep='first')
                
                print(f"\n{'='*80}")
                print("ROUND 3 SAMPLING COMPLETE")
                print(f"{'='*80}")
                print(f"\nTotal candidates: {len(candidates)}")
                print(f"  L1-focused: {len(l1_candidates) if l1_candidates is not None else 0}")
                print(f"  L2-focused: {len(l2_candidates) if l2_candidates is not None else 0}")
                print(f"  After dedup: {len(candidates)}")
                print(f"Average score: {candidates['sample_score'].mean():.2f}")
                
                # Re-save combined
                output_dir = Path('01_Data/06_manual_sets/02_sampling_candidates')
                output_path = output_dir / 'candidates_round3.csv'
                
                output_cols = [
                    'unit_id', 'sentence', 'role', 'prev', 'next',
                    'speakers', 'targets', 'source', 'date',
                    'sample_score', 'sampling_reason'
                ]
                # Add target columns
                if 'target_l1' in candidates.columns:
                    output_cols.insert(-2, 'target_l1')
                if 'target_l2' in candidates.columns:
                    output_cols.insert(-2, 'target_l2')
                
                available_cols = [col for col in output_cols if col in candidates.columns]
                candidates[available_cols].to_csv(output_path, index=False, encoding='utf-8-sig')
                
                print(f"\n[OK] Combined Round 3 candidates saved: {output_path}")
                
                print("\nNext steps:")
                print("1. python 08_prepare_candidates_for_annotation.py")
                print("2. python 05_validation_deepseek_annotation_async.py")
                print("3. Merge and verify gaps are filled")
        else:
            print(f"\n{'='*80}")
            print("ALL L1 TARGETS MET!")
            print(f"{'='*80}")
            print("\nProceed to full-scale annotation.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

