"""
Compare target L1 labels vs actual DeepSeek annotations
Analyze consistency and identify discrepancies
"""
import pandas as pd
import numpy as np
from collections import Counter

def compare_labels(annotation_csv: str, candidate_csv: str):
    """Compare target_l1 vs L1_label and generate detailed analysis"""
    
    # Load annotation results
    df_anno = pd.read_csv(annotation_csv, encoding='utf-8-sig')
    
    # Load original candidates with target_l1
    df_cand = pd.read_csv(candidate_csv, encoding='utf-8-sig')
    
    print("=" * 80)
    print("TARGET vs ANNOTATED LABEL COMPARISON")
    print("=" * 80)
    print(f"\nAnnotation samples: {len(df_anno)}")
    print(f"Candidate samples: {len(df_cand)}")
    
    # Check for duplicate IDs before merge
    anno_id_counts = df_anno['id'].value_counts()
    cand_id_counts = df_cand['id'].value_counts()
    
    if anno_id_counts.max() > 1:
        print(f"  Warning: Duplicate IDs in annotations: {(anno_id_counts > 1).sum()} duplicates")
        df_anno = df_anno.drop_duplicates(subset='id', keep='first')
        print(f"  After dedup: {len(df_anno)} samples")
    
    if cand_id_counts.max() > 1:
        print(f"  Warning: Duplicate IDs in candidates: {(cand_id_counts > 1).sum()} duplicates")
        df_cand = df_cand.drop_duplicates(subset='id', keep='first')
        print(f"  After dedup: {len(df_cand)} samples")
    
    # Merge on id (inner join to ensure matched samples only)
    df = df_anno.merge(
        df_cand[['id', 'target_l1', 'sample_score', 'score_reasons']], 
        on='id', 
        how='inner'
    )
    
    print(f"Merged samples: {len(df)}")
    
    # Check if both columns exist
    if 'target_l1' not in df.columns:
        print("Error: 'target_l1' column not found after merge")
        return
    
    if 'L1_label' not in df.columns:
        print("Error: 'L1_label' column not found")
        return
    
    # Overall match rate
    df['match'] = df['target_l1'] == df['L1_label']
    match_count = df['match'].sum()
    match_rate = match_count / len(df) * 100
    
    print(f"\n{'='*80}")
    print("OVERALL MATCH STATISTICS")
    print(f"{'='*80}")
    print(f"Matches: {match_count}/{len(df)} ({match_rate:.1f}%)")
    print(f"Mismatches: {len(df) - match_count}/{len(df)} ({100-match_rate:.1f}%)")
    
    # Match rate by target label
    print(f"\n{'='*80}")
    print("MATCH RATE BY TARGET L1 LABEL")
    print(f"{'='*80}")
    print(f"\n{'Target L1':<15} {'Total':>8} {'Matches':>10} {'Match Rate':>12}")
    print("-" * 80)
    
    match_by_target = df.groupby('target_l1').agg({
        'match': ['count', 'sum']
    })
    
    for target_l1 in sorted(df['target_l1'].unique()):
        target_df = df[df['target_l1'] == target_l1]
        total = len(target_df)
        matches = target_df['match'].sum()
        rate = matches / total * 100 if total > 0 else 0
        print(f"{target_l1:<15} {total:>8} {matches:>10} {rate:>11.1f}%")
    
    # Confusion matrix
    print(f"\n{'='*80}")
    print("CONFUSION MATRIX (Target L1 vs Annotated L1)")
    print(f"{'='*80}")
    
    confusion = pd.crosstab(
        df['target_l1'], 
        df['L1_label'], 
        margins=True,
        margins_name='Total'
    )
    print("\n", confusion)
    
    # Mismatched samples analysis
    print(f"\n{'='*80}")
    print("MISMATCHED SAMPLES ANALYSIS")
    print(f"{'='*80}")
    
    mismatch_df = df[~df['match']].copy()
    
    if len(mismatch_df) > 0:
        print(f"\nTotal mismatches: {len(mismatch_df)}")
        print(f"\nMost common mismatch patterns:")
        print(f"{'Target':>10} -> {'Annotated':<10} {'Count':>8}")
        print("-" * 80)
        
        mismatch_patterns = mismatch_df.groupby(['target_l1', 'L1_label']).size()
        for (target, annotated), count in mismatch_patterns.sort_values(ascending=False).head(15).items():
            print(f"{target:>10} -> {annotated:<10} {count:>8}")
        
        # Analyze confidence for mismatched samples
        if 'confidence' in mismatch_df.columns:
            print(f"\n{'='*80}")
            print("CONFIDENCE DISTRIBUTION FOR MISMATCHES")
            print(f"{'='*80}")
            conf_dist = mismatch_df['confidence'].value_counts()
            for conf, count in conf_dist.items():
                rate = count / len(mismatch_df) * 100
                print(f"  {conf}: {count} ({rate:.1f}%)")
        
        # Sample some mismatches for review
        print(f"\n{'='*80}")
        print("SAMPLE MISMATCHES (First 5)")
        print(f"{'='*80}")
        
        for idx, row in mismatch_df.head(5).iterrows():
            print(f"\nSample {idx + 1}:")
            print(f"  Target: {row['target_l1']}")
            print(f"  Annotated: {row['L1_label']}")
            print(f"  Confidence: {row.get('confidence', 'N/A')}")
            print(f"  Text: {str(row['sentence'])[:100]}...")
            if 'L1_reasoning' in row:
                print(f"  Reasoning: {str(row['L1_reasoning'])[:150]}...")
    
    # Matched samples - confidence check
    print(f"\n{'='*80}")
    print("MATCHED SAMPLES - CONFIDENCE DISTRIBUTION")
    print(f"{'='*80}")
    
    match_df = df[df['match']].copy()
    if len(match_df) > 0 and 'confidence' in match_df.columns:
        conf_dist = match_df['confidence'].value_counts()
        for conf, count in conf_dist.items():
            rate = count / len(match_df) * 100
            print(f"  {conf}: {count} ({rate:.1f}%)")
    
    # L2 labels analysis
    print(f"\n{'='*80}")
    print("L2 LABEL STATISTICS")
    print(f"{'='*80}")
    
    all_l2 = []
    for l2_str in df['L2_labels'].dropna():
        if l2_str and l2_str != '':
            labels = [l.strip() for l in str(l2_str).split('|') if l.strip()]
            all_l2.extend(labels)
    
    l2_counts = Counter(all_l2)
    avg_l2 = len(all_l2) / len(df)
    
    print(f"Total L2 instances: {len(all_l2)}")
    print(f"Average L2 per sample: {avg_l2:.2f}")
    print(f"\nL2 distribution (Top 10):")
    for label, count in l2_counts.most_common(10):
        rate = count / len(df) * 100
        print(f"  {label}: {count} ({rate:.1f}%)")
    
    # Export mismatch analysis
    output_path = annotation_csv.replace('.csv', '_mismatch_analysis.csv')
    if len(mismatch_df) > 0:
        mismatch_cols = ['id', 'target_l1', 'L1_label', 'confidence', 'sentence', 
                        'L1_reasoning', 'L2_labels', 'candidate', 'party', 'source_type',
                        'sample_score', 'score_reasons']
        available_cols = [col for col in mismatch_cols if col in mismatch_df.columns]
        mismatch_df[available_cols].to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n[OK] Mismatch details saved to: {output_path}")
    
    # Summary statistics
    summary = {
        'total_samples': len(df),
        'matches': int(match_count),
        'mismatches': int(len(df) - match_count),
        'match_rate': float(match_rate),
        'match_rate_by_target': {},
        'mismatch_patterns': {}
    }
    
    for target_l1 in sorted(df['target_l1'].dropna().unique()):
        target_df = df[df['target_l1'] == target_l1]
        total = len(target_df)
        matches = target_df['match'].sum()
        rate = matches / total * 100 if total > 0 else 0
        summary['match_rate_by_target'][target_l1] = {
            'total': int(total),
            'matches': int(matches),
            'rate': float(rate)
        }
    
    if len(mismatch_df) > 0:
        mismatch_patterns = mismatch_df.groupby(['target_l1', 'L1_label']).size()
        for (target, annotated), count in mismatch_patterns.items():
            key = f"{target}->{annotated}"
            summary['mismatch_patterns'][key] = int(count)
    
    summary_path = annotation_csv.replace('.csv', '_comparison_summary.json')
    import json
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"[OK] Summary statistics saved to: {summary_path}")
    
    return summary

def main():
    annotation_csv = '01_Data/06_manual_sets/03_results/04_tierC_annotations.csv'
    candidate_csv = '01_Data/06_manual_sets/02_sampling_candidates/candidates_tier_C_ready.csv'
    
    print("\n")
    
    try:
        compare_labels(annotation_csv, candidate_csv)
        
        print("\n" + "=" * 80)
        print("COMPARISON COMPLETE")
        print("=" * 80)
        print("\nKey insights:")
        print("1. High match rate (>70%) indicates good keyword selection")
        print("2. Low match rate may require keyword refinement")
        print("3. Check mismatch patterns for systematic issues")
        print("4. Review low-confidence mismatches manually")
        
    except FileNotFoundError:
        print(f"Error: File not found: {annotation_csv}")
        print("Please run annotation first:")
        print("  python 05_validation_deepseek_annotation_async.py")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

