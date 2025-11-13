"""
Compare Round 2 target vs annotated, then merge all batches
"""
import pandas as pd
import json
from collections import Counter

def compare_round2():
    """Compare Round 2 target vs annotated labels"""
    
    anno_path = '01_Data/06_manual_sets/03_results/06_round2_annotations.csv'
    cand_path = '01_Data/06_manual_sets/02_sampling_candidates/candidates_round2_ready.csv'
    
    df_anno = pd.read_csv(anno_path, encoding='utf-8-sig')
    df_cand = pd.read_csv(cand_path, encoding='utf-8-sig')
    
    print("=" * 80)
    print("ROUND 2: TARGET vs ANNOTATED COMPARISON")
    print("=" * 80)
    
    # Dedup
    df_anno = df_anno.drop_duplicates(subset='id', keep='first')
    df_cand = df_cand.drop_duplicates(subset='id', keep='first')
    
    print(f"Annotation samples: {len(df_anno)}")
    print(f"Candidate samples: {len(df_cand)}")
    
    # Merge
    df = df_anno.merge(df_cand[['id', 'target_l1']], on='id', how='inner')
    print(f"Matched samples: {len(df)}")
    
    # Compare
    df['match'] = df['target_l1'] == df['L1_label']
    match_count = df['match'].sum()
    match_rate = match_count / len(df) * 100
    
    print(f"\nOverall match rate: {match_count}/{len(df)} ({match_rate:.1f}%)")
    
    # By target label
    print(f"\n{'Target L1':<15} {'Total':>8} {'Matches':>10} {'Match Rate':>12}")
    print("-" * 80)
    
    for target_l1 in sorted(df['target_l1'].unique()):
        target_df = df[df['target_l1'] == target_l1]
        total = len(target_df)
        matches = target_df['match'].sum()
        rate = matches / total * 100 if total > 0 else 0
        print(f"{target_l1:<15} {total:>8} {matches:>10} {rate:>11.1f}%")
    
    # Actual L1 distribution from Round 2
    print(f"\n{'='*80}")
    print("ROUND 2: ACTUAL L1 DISTRIBUTION")
    print(f"{'='*80}")
    
    l1_dist = df_anno['L1_label'].value_counts()
    print(f"\n{'Label':<15} {'Count':>10}")
    print("-" * 80)
    for label, count in sorted(l1_dist.items()):
        print(f"{label:<15} {count:>10}")
    
    return df_anno

def merge_all_batches():
    """Merge all three batches: validation + tierC + round2"""
    
    print("\n" + "=" * 80)
    print("MERGING ALL ANNOTATION BATCHES")
    print("=" * 80)
    
    # Load all batches
    batch1 = pd.read_csv('01_Data/06_manual_sets/03_results/03_deepseek_annotations.csv', encoding='utf-8-sig')
    batch2 = pd.read_csv('01_Data/06_manual_sets/03_results/04_tierC_annotations.csv', encoding='utf-8-sig')
    batch3 = pd.read_csv('01_Data/06_manual_sets/03_results/06_round2_annotations.csv', encoding='utf-8-sig')
    
    batch1['batch'] = 'validation'
    batch2['batch'] = 'tierC'
    batch3['batch'] = 'round2'
    
    print(f"\nBatch 1 (validation): {len(batch1)}")
    print(f"Batch 2 (tierC): {len(batch2)}")
    print(f"Batch 3 (round2): {len(batch3)}")
    
    # Dedup within each batch
    batch1 = batch1.drop_duplicates(subset='id', keep='first')
    batch2 = batch2.drop_duplicates(subset='id', keep='first')
    batch3 = batch3.drop_duplicates(subset='id', keep='first')
    
    print(f"\nAfter dedup:")
    print(f"  Batch 1: {len(batch1)}")
    print(f"  Batch 2: {len(batch2)}")
    print(f"  Batch 3: {len(batch3)}")
    
    # Check overlaps
    ids1 = set(batch1['id'])
    ids2 = set(batch2['id'])
    ids3 = set(batch3['id'])
    
    overlap_12 = ids1 & ids2
    overlap_13 = ids1 & ids3
    overlap_23 = ids2 & ids3
    
    if overlap_12 or overlap_13 or overlap_23:
        print(f"\nOverlaps detected:")
        if overlap_12: print(f"  Batch1-Batch2: {len(overlap_12)}")
        if overlap_13: print(f"  Batch1-Batch3: {len(overlap_13)}")
        if overlap_23: print(f"  Batch2-Batch3: {len(overlap_23)}")
        print(f"  Keeping first occurrence")
        
        # Remove overlaps (keep batch1 > batch2 > batch3 priority)
        batch2 = batch2[~batch2['id'].isin(ids1)]
        batch3 = batch3[~batch3['id'].isin(ids1 | set(batch2['id']))]
    
    # Merge all
    merged = pd.concat([batch1, batch2, batch3], ignore_index=True)
    
    print(f"\n{'='*80}")
    print(f"FINAL MERGED DATASET")
    print(f"{'='*80}")
    print(f"Total samples: {len(merged)}")
    print(f"  Batch 1: {len(batch1)}")
    print(f"  Batch 2: {len(batch2)}")
    print(f"  Batch 3: {len(batch3)}")
    
    # Save
    output_path = '01_Data/06_manual_sets/03_results/07_all_merged_annotations.csv'
    merged.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n[OK] Saved: {output_path}")
    
    # Analyze final statistics
    print(f"\n{'='*80}")
    print("FINAL L1 DISTRIBUTION")
    print(f"{'='*80}")
    
    l1_counts = merged['L1_label'].value_counts()
    l1_target = 100
    
    print(f"\n{'Label':<15} {'Count':>10} {'Target':>10} {'Gap':>10} {'Status':<10}")
    print("-" * 80)
    
    total_gap = 0
    # L1 only has 9 labels (L1-01 to L1-09)
    for label in sorted([f'L1-{i:02d}' for i in range(1, 10)]):
        count = l1_counts.get(label, 0)
        gap = max(0, l1_target - count)
        total_gap += gap
        status = 'OK' if gap == 0 else f'+{gap}'
        print(f"{label:<15} {count:>10} {l1_target:>10} {gap:>10} {status:<10}")
    
    print(f"\nTotal L1 gap: +{total_gap}")
    
    # L2 analysis
    print(f"\n{'='*80}")
    print("FINAL L2 DISTRIBUTION")
    print(f"{'='*80}")
    
    all_l2 = []
    for l2_str in merged['L2_labels'].dropna():
        if l2_str and l2_str != '':
            labels = [l.strip() for l in str(l2_str).split('|') if l.strip()]
            all_l2.extend(labels)
    
    l2_counts = Counter(all_l2)
    l2_target = 75
    
    print(f"\nTotal L2 instances: {len(all_l2)}")
    print(f"Average per sample: {len(all_l2)/len(merged):.2f}")
    
    print(f"\n{'Label':<15} {'Count':>10} {'Target':>10} {'Gap':>10} {'Status':<10}")
    print("-" * 80)
    
    total_l2_gap = 0
    for label in sorted([f'L2-{i:02d}' for i in range(1, 16)]):
        count = l2_counts.get(label, 0)
        gap = max(0, l2_target - count)
        total_l2_gap += gap
        status = 'OK' if gap == 0 else f'+{gap}'
        print(f"{label:<15} {count:>10} {l2_target:>10} {gap:>10} {status:<10}")
    
    print(f"\nTotal L2 gap: +{total_l2_gap} instances")
    
    # Save statistics
    summary = {
        'total_samples': int(len(merged)),
        'batch_counts': {
            'validation': int(len(batch1)),
            'tierC': int(len(batch2)),
            'round2': int(len(batch3))
        },
        'l1_distribution': {k: int(v) for k, v in l1_counts.items()},
        'l2_distribution': {k: int(v) for k, v in l2_counts.items()},
        'l2_avg_per_sample': float(len(all_l2) / len(merged)),
        'total_l1_gap': int(total_gap),
        'total_l2_gap': int(total_l2_gap),
        'estimated_samples_for_l2': int(total_l2_gap / (len(all_l2) / len(merged)))
    }
    
    stats_path = '01_Data/06_manual_sets/03_results/07_all_merged_statistics.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n[OK] Statistics saved: {stats_path}")
    
    return summary

def main():
    try:
        # Compare Round 2
        df_round2 = compare_round2()
        
        # Merge all
        summary = merge_all_batches()
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        
        if summary['total_l1_gap'] > 0:
            print(f"\nRemaining L1 gap: +{summary['total_l1_gap']} samples")
            print(f"Remaining L2 gap: +{summary['total_l2_gap']} instances")
            print(f"  (estimated ~{summary['estimated_samples_for_l2']} samples needed)")
            print("\nConsider running another sampling round if needed.")
        else:
            print("\nAll L1 targets met!")
            print("Proceed to full-scale annotation.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

