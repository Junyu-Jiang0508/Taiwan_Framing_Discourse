"""
Prepare sampling candidates for DeepSeek annotation
Add required columns: id, candidate, party, source_type
"""
import pandas as pd
import re

def prepare_candidates(input_csv: str, output_csv: str):
    """Add required columns to candidates file"""
    df = pd.read_csv(input_csv, encoding='utf-8-sig')
    
    print(f"Loading candidates: {len(df)} samples")
    
    # Add id column (use unit_id if available, otherwise create sequential)
    if 'unit_id' in df.columns:
        df['id'] = df['unit_id']
    else:
        df['id'] = range(1, len(df) + 1)
    
    # Infer candidate and party from source
    def infer_candidate_party(source):
        source = str(source).lower()
        
        if '01_ho' in source or 'hou' in source or '侯' in source:
            return '侯友宜', 'KMT'
        elif '02_ke' in source or '柯' in source:
            return '柯文哲', 'TPP'
        elif '03_lai' in source or 'lai' in source or '赖' in source or '賴' in source:
            return '赖清德', 'DPP'
        else:
            return 'Unknown', 'Unknown'
    
    if 'candidate' not in df.columns or 'party' not in df.columns:
        df[['candidate', 'party']] = df['source'].apply(
            lambda x: pd.Series(infer_candidate_party(x))
        )
    
    # Infer source_type
    def infer_source_type(source):
        source = str(source).lower()
        
        if 'x_datasets' in source or 'twitter' in source:
            return 'social_media'
        elif 'meeting' in source or 'conference' in source or '辩论' in source or '专访' in source:
            return 'conference'
        else:
            return 'news'
    
    if 'source_type' not in df.columns:
        df['source_type'] = df['source'].apply(infer_source_type)
    
    # Ensure date column exists
    if 'date' not in df.columns:
        df['date'] = ''
    
    # Reorder columns for annotation
    required_cols = ['id', 'candidate', 'party', 'source_type', 'date', 'sentence']
    optional_cols = ['role', 'prev', 'next', 'speakers', 'targets', 
                     'proc_symbolic_flag', 'event_phase', 'target_l1', 
                     'sample_score', 'score_reasons', 'unit_id', 'source']
    
    output_cols = required_cols + [col for col in optional_cols if col in df.columns]
    
    df_output = df[output_cols]
    
    # Save
    df_output.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    print(f"\nPrepared dataset:")
    print(f"  Samples: {len(df_output)}")
    print(f"  Candidates: {df_output['candidate'].value_counts().to_dict()}")
    print(f"  Parties: {df_output['party'].value_counts().to_dict()}")
    print(f"  Source types: {df_output['source_type'].value_counts().to_dict()}")
    print(f"\nSaved to: {output_csv}")
    
    return df_output

def main():
    # Round 3: final gap filling
    input_csv = '01_Data/06_manual_sets/02_sampling_candidates/candidates_round3.csv'
    output_csv = '01_Data/06_manual_sets/02_sampling_candidates/candidates_round3_ready.csv'
    
    print("=" * 80)
    print("PREPARING CANDIDATES FOR ANNOTATION")
    print("=" * 80)
    
    try:
        prepare_candidates(input_csv, output_csv)
        
        print("\n" + "=" * 80)
        print("READY FOR ANNOTATION")
        print("=" * 80)
        print("\nNext step: Run annotation with:")
        print("  python 05_validation_deepseek_annotation_async.py")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

