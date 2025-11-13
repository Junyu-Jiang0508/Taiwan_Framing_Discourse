"""
Apply wrangling_bert filtering to sentence-level units
Adapted to work with units CSV files from preprocessing
"""
import pandas as pd
import os
import glob
import time
import json
from pathlib import Path
import importlib.util

# Import the filtering stack from utils_wrangling_bert
spec = importlib.util.spec_from_file_location("wrangling_bert", "utils_wrangling_bert.py")
wrangling_bert_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrangling_bert_module)
TextFilterStack = wrangling_bert_module.TextFilterStack

def process_units_file(
    input_file: str,
    output_file: str,
    filter_stack: TextFilterStack,
    save_all_categories: bool = True
):
    """Process a single units CSV file"""
    df = pd.read_csv(input_file, encoding='utf-8-sig')
    
    print(f"  Original units: {len(df)}")
    print(f"  Unit types: {df['role'].value_counts().to_dict()}")
    
    # Apply filtering pipeline
    result_df, stats = filter_stack.filter_pipeline(
        df,
        text_column='sentence',  # Our main text column
        use_bm25=False,
        use_semantic=True,
        use_nli=True
    )
    
    # Separate by final status
    keep_df = result_df[result_df['final_status'] == 'KEEP']
    drop_df = result_df[result_df['final_status'] == 'DROP']
    review_df = result_df[result_df['final_status'] == 'REVIEW']
    
    print(f"  After filtering:")
    print(f"    KEEP: {len(keep_df)}")
    print(f"    DROP: {len(drop_df)}")
    print(f"    REVIEW: {len(review_df)}")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save KEEP samples to main output file
    keep_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    # Save DROP and REVIEW samples to separate files
    if save_all_categories:
        base_name = os.path.splitext(output_file)[0]
        
        if len(drop_df) > 0:
            drop_file = base_name + '_DROP.csv'
            drop_df.to_csv(drop_file, index=False, encoding='utf-8-sig')
        
        if len(review_df) > 0:
            review_file = base_name + '_REVIEW.csv'
            review_df.to_csv(review_file, index=False, encoding='utf-8-sig')
    
    # Update stats
    stats['input_file'] = input_file
    stats['output_file'] = output_file
    stats['original_count'] = len(df)
    
    return stats

def batch_process_units():
    """Process all units CSV files"""
    filter_stack = TextFilterStack()
    
    all_stats = []
    
    # Define input/output directories
    base_input_dir = '01_Data/02_processed_datasets'
    base_output_dir = '01_Data/03_filtered_datasets'
    
    datasets = [
        '01_news_datasets',
        '02_conference_datasets',
        '03_X_datasets'
    ]
    
    # Collect all units CSV files
    all_files = []
    for dataset in datasets:
        input_dir = os.path.join(base_input_dir, dataset)
        output_dir = os.path.join(base_output_dir, dataset)
        
        if not os.path.exists(input_dir):
            print(f"Directory not found: {input_dir}")
            continue
        
        units_files = glob.glob(os.path.join(input_dir, '*_units.csv'))
        all_files.extend([(f, output_dir) for f in units_files])
    
    total_files = len(all_files)
    print(f"Total files to process: {total_files}")
    print("-" * 60)
    
    start_time = time.time()
    file_times = []
    processed_files = 0
    
    for input_file, output_dir in all_files:
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, filename)
        
        file_start = time.time()
        print(f"\nProcessing [{processed_files + 1}/{total_files}]: {filename}")
        
        try:
            stats = process_units_file(
                input_file=input_file,
                output_file=output_file,
                filter_stack=filter_stack
            )
            
            file_time = time.time() - file_start
            file_times.append(file_time)
            processed_files += 1
            
            all_stats.append(stats)
            
            print(f"  ✓ Completed in {file_time:.1f}s")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Save overall statistics
    stats_output = os.path.join(base_output_dir, '_filtering_stats.json')
    
    # Convert numpy types to native Python types
    def convert_to_serializable(obj):
        import numpy as np
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        else:
            return obj
    
    all_stats_serializable = convert_to_serializable(all_stats)
    
    os.makedirs(os.path.dirname(stats_output), exist_ok=True)
    with open(stats_output, 'w', encoding='utf-8') as f:
        json.dump(all_stats_serializable, f, ensure_ascii=False, indent=2)
    
    # Print summary
    total_time = time.time() - start_time
    hours = int(total_time // 3600)
    minutes = int((total_time % 3600) // 60)
    seconds = int(total_time % 60)
    
    print("\n" + "=" * 60)
    print("BATCH PROCESSING COMPLETED!")
    print("=" * 60)
    print(f"Total files processed: {len(all_stats)}/{total_files}")
    
    if hours > 0:
        print(f"Total processing time: {hours}h {minutes}m {seconds}s")
    elif minutes > 0:
        print(f"Total processing time: {minutes}m {seconds}s")
    else:
        print(f"Total processing time: {seconds}s")
    
    if len(file_times) > 0:
        avg_time = sum(file_times) / len(file_times)
        print(f"Average time per file: {avg_time:.1f}s")
    
    total_original = sum(s.get('original_count', 0) for s in all_stats)
    total_keep = sum(s.get('final_KEEP', 0) for s in all_stats)
    total_drop = sum(s.get('final_DROP', 0) for s in all_stats)
    total_review = sum(s.get('final_REVIEW', 0) for s in all_stats)
    
    total_after_hard = sum(s.get('after_hard_rule_keep', 0) for s in all_stats)
    total_hard_dropped = total_original - total_after_hard
    
    print(f"\nOverall statistics:")
    print(f"  Total original units: {total_original}")
    print(f"\n  After preprocessing & hard rules:")
    print(f"    - Passed hard rules: {total_after_hard} ({total_after_hard/total_original*100:.1f}%)")
    print(f"    - Dropped by hard rules: {total_hard_dropped} ({total_hard_dropped/total_original*100:.1f}%)")
    print(f"\n  After semantic & NLI filtering:")
    print(f"    - KEEP: {total_keep} ({total_keep/total_after_hard*100:.1f}% of candidates, {total_keep/total_original*100:.1f}% of original)")
    print(f"    - DROP: {total_drop} ({total_drop/total_after_hard*100:.1f}% of candidates, {total_drop/total_original*100:.1f}% of original)")
    print(f"    - REVIEW: {total_review} ({total_review/total_after_hard*100:.1f}% of candidates, {total_review/total_original*100:.1f}% of original)")
    print(f"\n  Final retention rate: {total_keep/total_original*100:.1f}%")
    print(f"\nStatistics saved to: {stats_output}")
    print(f"Filtered datasets saved to: {base_output_dir}")
    print("=" * 60)

def main():
    batch_process_units()

if __name__ == "__main__":
    main()

