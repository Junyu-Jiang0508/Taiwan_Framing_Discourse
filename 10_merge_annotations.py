"""
Merge multiple annotation batches and analyze combined statistics
"""
import pandas as pd
import json
from pathlib import Path

def merge_annotations():
    """Merge validation set + tierC annotations"""
    
    # Load original validation set annotations
    anno1_path = '01_Data/06_manual_sets/03_results/03_deepseek_annotations.csv'
    anno1 = pd.read_csv(anno1_path, encoding='utf-8-sig')
    anno1['batch'] = 'validation_set'
    
    # Load tierC annotations
    anno2_path = '01_Data/06_manual_sets/03_results/04_tierC_annotations.csv'
    anno2 = pd.read_csv(anno2_path, encoding='utf-8-sig')
    anno2['batch'] = 'tierC_completion'
    
    print("=" * 80)
    print("MERGING ANNOTATIONS")
    print("=" * 80)
    print(f"\nBatch 1 (validation set): {len(anno1)} samples")
    print(f"Batch 2 (tierC): {len(anno2)} samples")
    
    # Check for duplicate IDs
    anno1_dedup = anno1.drop_duplicates(subset='id', keep='first')
    anno2_dedup = anno2.drop_duplicates(subset='id', keep='first')
    
    if len(anno1_dedup) < len(anno1):
        print(f"  Removed {len(anno1) - len(anno1_dedup)} duplicates from batch 1")
    if len(anno2_dedup) < len(anno2):
        print(f"  Removed {len(anno2) - len(anno2_dedup)} duplicates from batch 2")
    
    # Check for overlapping IDs between batches
    overlap = set(anno1_dedup['id']) & set(anno2_dedup['id'])
    if overlap:
        print(f"\n  Warning: {len(overlap)} overlapping IDs between batches")
        print(f"  Keeping batch 1 version for overlaps")
        anno2_dedup = anno2_dedup[~anno2_dedup['id'].isin(overlap)]
    
    # Merge
    merged = pd.concat([anno1_dedup, anno2_dedup], ignore_index=True)
    
    print(f"\nMerged total: {len(merged)} samples")
    print(f"  From batch 1: {len(anno1_dedup)}")
    print(f"  From batch 2: {len(anno2_dedup)}")
    
    # Save merged annotations
    output_path = '01_Data/06_manual_sets/03_results/05_merged_annotations.csv'
    merged.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\n[OK] Merged annotations saved to: {output_path}")
    
    # Analyze merged statistics
    print("\n" + "=" * 80)
    print("MERGED ANNOTATION STATISTICS")
    print("=" * 80)
    
    # L1 distribution
    l1_counts = merged['L1_label'].value_counts()
    print(f"\nL1 Label Distribution:")
    print(f"{'Label':<15} {'Count':>10} {'Percentage':>12}")
    print("-" * 80)
    for label, count in sorted(l1_counts.items()):
        percentage = count / len(merged) * 100
        print(f"{label:<15} {count:>10} {percentage:>11.1f}%")
    
    # L2 statistics
    all_l2 = []
    for l2_str in merged['L2_labels'].dropna():
        if l2_str and l2_str != '':
            labels = [l.strip() for l in str(l2_str).split('|') if l.strip()]
            all_l2.extend(labels)
    
    from collections import Counter
    l2_counts = Counter(all_l2)
    
    print(f"\nL2 Label Statistics:")
    print(f"  Total instances: {len(all_l2)}")
    print(f"  Average per sample: {len(all_l2) / len(merged):.2f}")
    print(f"  Unique labels: {len(l2_counts)}")
    
    print(f"\nL2 Distribution:")
    print(f"{'Label':<15} {'Count':>10} {'Percentage':>12}")
    print("-" * 80)
    for label, count in sorted(l2_counts.items()):
        percentage = count / len(merged) * 100
        print(f"{label:<15} {count:>10} {percentage:>11.1f}%")
    
    # Identify gaps (for next round)
    print("\n" + "=" * 80)
    print("REMAINING GAPS ANALYSIS")
    print("=" * 80)
    
    # C-tier targets
    l1_target = 100
    l2_target = 75
    
    l1_gaps = {}
    l2_gaps = {}
    
    for label in [f'L1-{i:02d}' for i in range(1, 16)]:
        current = l1_counts.get(label, 0)
        if current < l1_target:
            l1_gaps[label] = l1_target - current
    
    for label in [f'L2-{i:02d}' for i in range(1, 16)]:
        current = l2_counts.get(label, 0)
        if current < l2_target:
            l2_gaps[label] = l2_target - current
    
    print(f"\nL1 gaps (target: {l1_target} per label):")
    if l1_gaps:
        total_l1_gap = sum(l1_gaps.values())
        print(f"{'Label':<15} {'Current':>10} {'Gap':>10}")
        print("-" * 80)
        for label, gap in sorted(l1_gaps.items(), key=lambda x: x[1], reverse=True):
            current = l1_counts.get(label, 0)
            print(f"{label:<15} {current:>10} +{gap:>9}")
        print(f"\nTotal L1 gap: +{total_l1_gap} samples needed")
    else:
        print("  All L1 labels meet target!")
    
    print(f"\nL2 gaps (target: {l2_target} per label):")
    if l2_gaps:
        total_l2_gap = sum(l2_gaps.values())
        print(f"{'Label':<15} {'Current':>10} {'Gap':>10}")
        print("-" * 80)
        for label, gap in sorted(l2_gaps.items(), key=lambda x: x[1], reverse=True):
            current = l2_counts.get(label, 0)
            print(f"{label:<15} {current:>10} +{gap:>9}")
        print(f"\nTotal L2 gap: +{total_l2_gap} instances needed")
        avg_l2 = len(all_l2) / len(merged)
        estimated = int(total_l2_gap / avg_l2)
        print(f"Estimated samples needed: ~{estimated} (assuming {avg_l2:.2f} L2/sample)")
    else:
        print("  All L2 labels meet target!")
    
    # Convert numpy types to Python native types
    def convert_dict(d):
        return {k: int(v) if hasattr(v, 'item') else v for k, v in d.items()}
    
    # Save statistics
    summary = {
        'total_samples': int(len(merged)),
        'batch1_count': int(len(anno1_dedup)),
        'batch2_count': int(len(anno2_dedup)),
        'l1_distribution': convert_dict(l1_counts.to_dict()),
        'l2_distribution': convert_dict(dict(l2_counts)),
        'l2_avg_per_sample': float(len(all_l2) / len(merged)),
        'l1_gaps': convert_dict(l1_gaps),
        'l2_gaps': convert_dict(l2_gaps),
        'l1_target': int(l1_target),
        'l2_target': int(l2_target)
    }
    
    stats_path = '01_Data/06_manual_sets/03_results/05_merged_statistics.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print(f"\n[OK] Statistics saved to: {stats_path}")
    
    return summary

def main():
    print("\n")
    
    try:
        summary = merge_annotations()
        
        print("\n" + "=" * 80)
        print("MERGE COMPLETE")
        print("=" * 80)
        
        total_l1_gap = sum(summary['l1_gaps'].values())
        total_l2_gap = sum(summary['l2_gaps'].values())
        
        if total_l1_gap > 0 or total_l2_gap > 0:
            print("\nNext steps:")
            print("1. Review gap analysis above")
            print("2. Run targeted sampling for remaining gaps:")
            print("   python 11_targeted_sampling_for_gaps.py")
            print("3. Annotate additional samples")
            print("4. Repeat merge and check until all gaps filled")
        else:
            print("\nAll targets met! Proceed to:")
            print("1. Build ICL example library")
            print("2. Run full-scale GPT-4o annotation")
            print("3. Train verifier model")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

