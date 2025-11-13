"""
Convert JSONL sentence-level units to CSV format for wrangling_bert.py
Preserves all metadata for downstream processing
"""
import json
import pandas as pd
from pathlib import Path
import os

def convert_jsonl_to_csv(jsonl_path: str, csv_path: str):
    """Convert JSONL units file to CSV format"""
    units = []
    
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            unit = json.loads(line)
            
            # Flatten structure for CSV
            row = {
                'doc_id': unit['doc_id'],
                'unit_id': unit['unit_id'],
                'sentence': unit['text'],  # Main text column for wrangling_bert
                'role': unit['role'],
                'prev': unit['prev'],
                'next': unit['next'],
                'speakers': json.dumps(unit['speakers'], ensure_ascii=False),
                'targets': json.dumps(unit['targets'], ensure_ascii=False),
                'proc_symbolic_flag': unit['proc_symbolic_flag'],
                'event_phase': unit['event_phase'],
                'unit_hash': unit['unit_hash'],
                'unit_dup_flag': unit['unit_dup_flag'],
                'source': unit['source_meta']['source'],
                'date': unit['source_meta']['date']
            }
            units.append(row)
    
    df = pd.DataFrame(units)
    
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    return len(df)

def batch_convert():
    """Convert all JSONL files in processed_datasets to CSV"""
    base_dir = Path("01_Data/02_processed_datasets")
    
    datasets = [
        "01_news_datasets",
        "02_conference_datasets",
        "03_X_datasets"
    ]
    
    total_units = 0
    total_files = 0
    
    for dataset in datasets:
        dataset_dir = base_dir / dataset
        
        if not dataset_dir.exists():
            print(f"Directory not found: {dataset_dir}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Converting {dataset}")
        print('='*60)
        
        for jsonl_file in dataset_dir.glob("*_units.jsonl"):
            csv_file = jsonl_file.with_name(jsonl_file.stem.replace('_units', '_units') + '.csv')
            
            print(f"\n  {jsonl_file.name}")
            print(f"  → {csv_file.name}")
            
            try:
                count = convert_jsonl_to_csv(str(jsonl_file), str(csv_file))
                print(f"  ✓ Converted {count} units")
                total_units += count
                total_files += 1
            except Exception as e:
                print(f"  ✗ Error: {e}")
    
    print(f"\n{'='*60}")
    print("Conversion complete!")
    print('='*60)
    print(f"Total files converted: {total_files}")
    print(f"Total units converted: {total_units}")
    print(f"\nCSV files are ready for 01_wrangling_bert.py processing")

if __name__ == "__main__":
    batch_convert()

