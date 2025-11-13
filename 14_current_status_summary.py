"""
Current status summary - What do we have and what's missing?
"""
import pandas as pd
from collections import Counter

def summarize_current_status():
    """Generate a clear summary of current annotation status"""
    
    csv_path = '01_Data/06_manual_sets/03_results/05_merged_annotations.csv'
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    
    print("=" * 80)
    print("CURRENT ANNOTATION STATUS SUMMARY")
    print("=" * 80)
    print(f"\nTotal gold-standard samples: {len(df)}")
    
    # L1 Analysis
    l1_counts = df['L1_label'].value_counts()
    l1_target = 100
    
    print("\n" + "=" * 80)
    print("L1 LABELS (9 valid labels, target: 100 each)")
    print("=" * 80)
    
    met_l1 = []
    near_l1 = []
    gap_l1 = []
    
    for i in range(1, 10):
        label = f'L1-{i:02d}'
        count = l1_counts.get(label, 0)
        gap = max(0, l1_target - count)
        
        if count >= l1_target:
            met_l1.append((label, count, count - l1_target))
        elif count >= 80:
            near_l1.append((label, count, gap))
        else:
            gap_l1.append((label, count, gap))
    
    print(f"\n[OK] MET TARGET ({len(met_l1)}/9):")
    for label, count, surplus in met_l1:
        print(f"  {label}: {count} (+{surplus})")
    
    print(f"\n[~] CLOSE TO TARGET (80-99) ({len(near_l1)}/9):")
    for label, count, gap in near_l1:
        print(f"  {label}: {count} (need +{gap})")
    
    print(f"\n[X] BELOW TARGET (<80) ({len(gap_l1)}/9):")
    for label, count, gap in sorted(gap_l1, key=lambda x: x[2], reverse=True):
        print(f"  {label}: {count} (need +{gap})")
    
    total_l1_gap = sum(gap for _, _, gap in near_l1 + gap_l1)
    print(f"\n[STAT] Total L1 gap: +{total_l1_gap} samples needed")
    
    # Invalid L1 check
    invalid_l1 = [(label, l1_counts[label]) for label in l1_counts.index 
                  if pd.notna(label) and label.startswith('L1-') and int(label.split('-')[1]) > 9]
    
    if invalid_l1:
        print(f"\n[WARNING] INVALID L1 LABELS DETECTED:")
        for label, count in invalid_l1:
            print(f"  {label}: {count} samples (DeepSeek error - need to fix prompt)")
    
    # L2 Analysis
    all_l2 = []
    for l2_str in df['L2_labels'].dropna():
        if l2_str and l2_str != '':
            labels = [l.strip() for l in str(l2_str).split('|') if l.strip()]
            all_l2.extend(labels)
    
    l2_counts = Counter(all_l2)
    l2_target = 75
    
    print("\n" + "=" * 80)
    print("L2 LABELS (15 labels, target: 75 each)")
    print("=" * 80)
    print(f"\nTotal L2 instances: {len(all_l2)}")
    print(f"Average per sample: {len(all_l2)/len(df):.2f}")
    
    met_l2 = []
    gap_l2 = []
    
    for i in range(1, 16):
        label = f'L2-{i:02d}'
        count = l2_counts.get(label, 0)
        gap = max(0, l2_target - count)
        
        if count >= l2_target:
            met_l2.append((label, count, count - l2_target))
        else:
            gap_l2.append((label, count, gap))
    
    print(f"\n[OK] MET TARGET ({len(met_l2)}/15):")
    for label, count, surplus in sorted(met_l2, key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {label}: {count} (+{surplus})")
    if len(met_l2) > 5:
        print(f"  ... and {len(met_l2)-5} more")
    
    print(f"\n[X] BELOW TARGET ({len(gap_l2)}/15):")
    for label, count, gap in sorted(gap_l2, key=lambda x: x[2], reverse=True):
        print(f"  {label}: {count} (need +{gap})")
    
    total_l2_gap = sum(gap for _, _, gap in gap_l2)
    avg_l2_per_sample = len(all_l2) / len(df)
    estimated_samples = int(total_l2_gap / avg_l2_per_sample)
    
    print(f"\n[STAT] Total L2 gap: +{total_l2_gap} instances")
    print(f"[STAT] Estimated samples needed: ~{estimated_samples} (at {avg_l2_per_sample:.2f} L2/sample)")
    
    # Final recommendation
    print("\n" + "=" * 80)
    print("SUMMARY & RECOMMENDATION")
    print("=" * 80)
    
    recommendation = max(total_l1_gap, estimated_samples)
    
    print(f"\n[CURRENT] Dataset: {len(df)} samples")
    print(f"   - L1 met: {len(met_l1)}/9 ({len(met_l1)/9*100:.0f}%)")
    print(f"   - L2 met: {len(met_l2)}/15 ({len(met_l2)/15*100:.0f}%)")
    
    print(f"\n[GAPS] Remaining:")
    print(f"   - L1: +{total_l1_gap} samples")
    print(f"   - L2: ~{estimated_samples} samples")
    
    print(f"\n[RECOMMENDATION]:")
    print(f"   Round 3 should add: ~{recommendation} samples")
    
    print(f"\n[PREDICTION] After Round 3:")
    predicted_total = len(df) + recommendation
    print(f"   Total samples: ~{predicted_total}")
    print(f"   L1 coverage: Expected 9/9 or close")
    print(f"   L2 coverage: Expected 15/15 or close")
    
    print(f"\n[READY] Round 3 preparation:")
    print(f"   [OK] candidates_round3.csv generated (285 candidates)")
    print(f"   [OK] Includes both L1 and L2 gap targets")
    print(f"   [TODO] Run: python 08_prepare_candidates_for_annotation.py")
    print(f"   [TODO] Then: python 05_validation_deepseek_annotation_async.py")

def main():
    try:
        summarize_current_status()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

