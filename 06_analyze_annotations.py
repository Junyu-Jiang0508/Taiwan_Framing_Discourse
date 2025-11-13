"""
Analyze annotation results and generate statistics
"""
import pandas as pd
from collections import Counter
import json

def analyze_annotations(csv_path: str):
    """Analyze L1 and L2 label distributions"""
    
    # Load data
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    
    print("=" * 80)
    print("ANNOTATION STATISTICS")
    print("=" * 80)
    print(f"\nTotal samples: {len(df)}")
    
    # Check for errors
    error_count = df['error'].notna().sum() if 'error' in df.columns else 0
    if error_count > 0:
        print(f"Samples with errors: {error_count}")
    
    print(f"Successfully annotated: {len(df) - error_count}")
    
    # L1 Label Distribution
    print("\n" + "=" * 80)
    print("L1 LABEL DISTRIBUTION (Single-choice)")
    print("=" * 80)
    
    l1_counts = df['L1_label'].value_counts()
    print(f"\nTotal L1 labels found: {l1_counts.sum()}")
    print(f"Unique L1 labels: {len(l1_counts)}")
    print("\nDistribution:")
    print("-" * 80)
    print(f"{'Label':<15} {'Count':>10} {'Percentage':>12}")
    print("-" * 80)
    
    for label, count in l1_counts.items():
        if pd.notna(label):
            percentage = count / len(df) * 100
            print(f"{label:<15} {count:>10} {percentage:>11.1f}%")
    
    # L2 Labels Distribution
    print("\n" + "=" * 80)
    print("L2 LABEL DISTRIBUTION (Multi-choice)")
    print("=" * 80)
    
    # Parse L2 labels (they are pipe-separated)
    all_l2_labels = []
    l2_combinations = []
    
    for l2_str in df['L2_labels'].dropna():
        if l2_str and l2_str != '':
            labels = [l.strip() for l in str(l2_str).split('|') if l.strip()]
            all_l2_labels.extend(labels)
            l2_combinations.append(tuple(sorted(labels)))
    
    l2_counts = Counter(all_l2_labels)
    
    print(f"\nTotal L2 label instances: {len(all_l2_labels)}")
    print(f"Unique L2 labels: {len(l2_counts)}")
    print(f"Average L2 labels per sample: {len(all_l2_labels) / len(df):.2f}")
    
    print("\nDistribution:")
    print("-" * 80)
    print(f"{'Label':<15} {'Count':>10} {'Percentage':>12}")
    print("-" * 80)
    
    for label, count in sorted(l2_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / len(df) * 100
        print(f"{label:<15} {count:>10} {percentage:>11.1f}%")
    
    # L2 Combinations
    print("\n" + "=" * 80)
    print("L2 LABEL COMBINATIONS (Top 10)")
    print("=" * 80)
    
    combo_counts = Counter(l2_combinations)
    print(f"\nTotal unique combinations: {len(combo_counts)}")
    print("\nTop 10 combinations:")
    print("-" * 80)
    print(f"{'Combination':<50} {'Count':>10}")
    print("-" * 80)
    
    for combo, count in combo_counts.most_common(10):
        combo_str = ' + '.join(combo) if combo else 'None'
        print(f"{combo_str:<50} {count:>10}")
    
    # Confidence Distribution
    print("\n" + "=" * 80)
    print("CONFIDENCE DISTRIBUTION")
    print("=" * 80)
    
    confidence_counts = df['confidence'].value_counts()
    print("\nDistribution:")
    print("-" * 80)
    print(f"{'Confidence':<15} {'Count':>10} {'Percentage':>12}")
    print("-" * 80)
    
    for conf, count in confidence_counts.items():
        if pd.notna(conf):
            percentage = count / len(df) * 100
            print(f"{conf:<15} {count:>10} {percentage:>11.1f}%")
    
    # Cross-tabulation: L1 vs Candidate
    print("\n" + "=" * 80)
    print("L1 LABELS BY CANDIDATE")
    print("=" * 80)
    
    if 'candidate' in df.columns:
        crosstab = pd.crosstab(df['L1_label'], df['candidate'], margins=True)
        print("\n", crosstab)
    
    # Cross-tabulation: L1 vs Party
    print("\n" + "=" * 80)
    print("L1 LABELS BY PARTY")
    print("=" * 80)
    
    if 'party' in df.columns:
        crosstab = pd.crosstab(df['L1_label'], df['party'], margins=True)
        print("\n", crosstab)
    
    # Source type distribution
    print("\n" + "=" * 80)
    print("LABELS BY SOURCE TYPE")
    print("=" * 80)
    
    if 'source_type' in df.columns:
        source_dist = df['source_type'].value_counts()
        print("\nSample distribution by source:")
        for source, count in source_dist.items():
            percentage = count / len(df) * 100
            print(f"  {source}: {count} ({percentage:.1f}%)")
    
    # Save summary to JSON
    summary = {
        'total_samples': len(df),
        'error_count': int(error_count),
        'l1_distribution': l1_counts.to_dict(),
        'l2_distribution': dict(l2_counts),
        'l2_avg_per_sample': len(all_l2_labels) / len(df),
        'confidence_distribution': confidence_counts.to_dict(),
        'top_10_l2_combinations': [
            {'labels': list(combo), 'count': count} 
            for combo, count in combo_counts.most_common(10)
        ]
    }
    
    output_json = csv_path.replace('.csv', '_statistics.json')
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 80)
    print(f"Statistics saved to: {output_json}")
    print("=" * 80)

def main():
    csv_path = '01_Data/06_manual_sets/03_results/03_deepseek_annotations.csv'
    
    try:
        analyze_annotations(csv_path)
    except FileNotFoundError:
        print(f"Error: File not found: {csv_path}")
        print("Please run annotation first using:")
        print("  python 05_validation_deepseek_annotation.py")
        print("  or")
        print("  python 05_validation_deepseek_annotation_async.py")
    except Exception as e:
        print(f"Error analyzing annotations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

