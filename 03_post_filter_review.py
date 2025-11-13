"""
Post-filtering script for REVIEW subset
Refine REVIEW samples into: REVIEW_KEEP, REVIEW_CONTEXT, REVIEW_DROP
Based on semantic similarity, NLI scores, and domain-specific cues
"""
import pandas as pd
import os
import glob
import json
from pathlib import Path

# Actor terms (人名/党派/阵营)
ACTOR_TERMS = {
    '赖清德', '賴清德', '萧美琴', '蕭美琴',
    '侯友宜', '赵少康', '趙少康',
    '柯文哲', '吴欣盈', '吳欣盈',
    '民进党', '民進黨', '国民党', '國民黨', '民众党', '民眾黨',
    '绿营', '綠營', '蓝营', '藍營', '白营', '白營',
    'DPP', 'KMT', 'TPP'
}

# Institution/Issue terms (制度/议题)
INST_TERMS = {
    '两岸', '兩岸', '台海', '主权', '主權', '国家认同', '國家認同',
    '一国两制', '一國兩制', '正名',
    '国防', '國防', '兵役', '军演', '軍演', '导弹', '導彈',
    '经济', '經濟', '通膨', '物价', '物價', '房价', '房價',
    '核能', '核四', '再生能源', '风电', '風電', '光电', '光電',
    '社宅', '年金', '健保', '长照', '長照', '托育',
    '居住正义', '居住正義', '性别平权', '性別平權', '婚姻平权', '婚姻平權',
    '立法院', '立委', '三读', '三讀', '质询', '質詢',
    'ECFA', '修宪', '修憲'
}

# Procedural/Symbolic cues (程序/象征)
PROC_SYMBOLIC = {
    '政见发表会', '政見發表會', '辩论', '辯論',
    '三读', '三讀', '表决', '表決', '院会', '院會',
    '造势', '造勢', '扫街', '掃街', '拜票',
    '号次', '號次', '抽签', '抽籤',
    '升旗', '挥舞国旗', '揮舞國旗', '国旗', '國旗',
    '集会', '集會', '游行', '遊行', '动员', '動員',
    '协商', '協商', '版本', '草案', '投票'
}

# Scarce L1/L2 anchor terms (稀缺标签锚词)
SCARCE_L1_L2 = {
    # Economic frames
    '经济发展', '經濟發展', '经济成长', '經濟成長', '产业升级', '產業升級',
    '贸易', '貿易', '投资', '投資', '就业', '就業', '薪资', '薪資',
    # Social justice
    '社会正义', '社會正義', '居住正义', '居住正義', '分配正义', '分配正義',
    '世代正义', '世代正義', '转型正义', '轉型正義',
    # National symbols
    '国格', '國格', '国家尊严', '國家尊嚴', '主权', '主權',
    # Collective memory
    '集体记忆', '集體記憶', '共同体', '共同體', '历史正义', '歷史正義',
    '二二八', '白色恐怖', '威权', '威權',
    # National pride
    '民族自豪', '台湾价值', '臺灣價值', '民主', '自由', '人权', '人權'
}

# Strong blacklist (强黑名单)
STRONG_BLACKLIST = {
    '抽奖', '抽獎', '中奖', '中獎', '博彩', '賭博',
    '导购', '導購', '团购', '團購', '优惠', '優惠',
    '促销', '促銷', '打折', '特价', '特價',
    '彩票', '彩券', '刮刮乐', '刮刮樂'
}

# Taiwan context markers
TAIWAN_MARKERS = {
    '台湾', '臺灣', '台灣', '中华民国', '中華民國',
    '台北', '臺北', '高雄', '台中', '臺中',
    '本土', '在地', '我国', '我國'
}

def check_actor(text: str) -> bool:
    """Check if text contains actor terms"""
    return any(term in text for term in ACTOR_TERMS)

def check_inst(text: str) -> bool:
    """Check if text contains institution/issue terms"""
    return any(term in text for term in INST_TERMS)

def check_proc_symbolic(text: str) -> bool:
    """Check if text contains procedural/symbolic cues"""
    return any(term in text for term in PROC_SYMBOLIC)

def check_scarce_l1_l2(text: str) -> bool:
    """Check if text contains scarce L1/L2 anchor terms"""
    return any(term in text for term in SCARCE_L1_L2)

def check_blacklist(text: str) -> bool:
    """Check if text hits strong blacklist"""
    return any(term in text for term in STRONG_BLACKLIST)

def check_taiwan_context(text: str) -> bool:
    """Check if text has Taiwan context markers"""
    return any(term in text for term in TAIWAN_MARKERS)

def classify_review(row: pd.Series) -> str:
    """
    Classify REVIEW samples into:
    - REVIEW_KEEP: Enter annotation pipeline
    - REVIEW_CONTEXT: Keep as context/sample padding
    - REVIEW_DROP: Actually drop
    """
    text = str(row.get('sentence', ''))
    s = row.get('semantic_similarity', 0.0)
    m = row.get('nli_score', 0.0)  # NLI margin score
    
    # Convert to float if needed
    try:
        s = float(s)
    except:
        s = 0.0
    
    try:
        m = float(m)
    except:
        m = 0.0
    
    # Check domain cues
    has_actor = check_actor(text)
    has_inst = check_inst(text)
    has_proc_symbolic = check_proc_symbolic(text)
    has_scarce = check_scarce_l1_l2(text)
    has_blacklist = check_blacklist(text)
    has_taiwan = check_taiwan_context(text)
    
    # REVIEW_DROP conditions (most restrictive first)
    if has_blacklist and not (has_actor or has_inst):
        return 'REVIEW_DROP'
    
    # Check for off-domain: mentions US election but no Taiwan context
    if ('美国' in text or '美國' in text) and '选举' in text and not has_taiwan:
        if not (has_actor or has_inst):
            return 'REVIEW_DROP'
    
    # REVIEW_KEEP conditions (any one satisfied)
    if s >= 0.40:  # keep_th - 0.05 buffer
        return 'REVIEW_KEEP'
    
    if has_actor or has_inst:
        return 'REVIEW_KEEP'
    
    if m >= 0.10:  # NLI slightly positive
        return 'REVIEW_KEEP'
    
    if has_scarce:
        return 'REVIEW_KEEP'
    
    # REVIEW_CONTEXT conditions (any one satisfied)
    if has_proc_symbolic:
        return 'REVIEW_CONTEXT'
    
    if 0.30 <= s < 0.40 and (has_actor or has_inst):
        return 'REVIEW_CONTEXT'
    
    if 0.0 < m < 0.10:
        return 'REVIEW_CONTEXT'
    
    # Default: REVIEW_DROP
    return 'REVIEW_DROP'

def process_review_file(review_file: str, main_file: str, output_dir: str) -> dict:
    """
    Process a REVIEW file and merge qualified samples back to main file
    
    Args:
        review_file: Path to *_REVIEW.csv file
        main_file: Path to main *_units.csv file (contains KEEP samples)
        output_dir: Output directory
    """
    print(f"\nProcessing: {os.path.basename(review_file)}")
    
    # Read REVIEW file
    review_df = pd.read_csv(review_file, encoding='utf-8-sig')
    review_count = len(review_df)
    print(f"  REVIEW samples: {review_count}")
    
    # Read main file (KEEP samples)
    if os.path.exists(main_file):
        main_df = pd.read_csv(main_file, encoding='utf-8-sig')
        keep_count = len(main_df)
        print(f"  KEEP samples in main file: {keep_count}")
    else:
        print(f"  Warning: Main file not found: {main_file}")
        main_df = pd.DataFrame()
        keep_count = 0
    
    # Classify REVIEW samples
    review_df['refined_status'] = review_df.apply(classify_review, axis=1)
    
    refined_counts = review_df['refined_status'].value_counts().to_dict()
    print(f"  REVIEW classification: {refined_counts}")
    
    # Split REVIEW by refined status
    review_keep_df = review_df[review_df['refined_status'] == 'REVIEW_KEEP'].copy()
    review_context_df = review_df[review_df['refined_status'] == 'REVIEW_CONTEXT'].copy()
    review_drop_df = review_df[review_df['refined_status'] == 'REVIEW_DROP'].copy()
    
    # Remove refined_status column before merging (keep original structure)
    review_keep_df = review_keep_df.drop(columns=['refined_status'])
    
    # Combine KEEP + REVIEW_KEEP for final main file
    final_main_df = pd.concat([main_df, review_keep_df], ignore_index=True)
    
    print(f"\n  Final distribution:")
    print(f"    Main file (KEEP + REVIEW_KEEP): {len(final_main_df)}")
    print(f"      Original KEEP: {keep_count}")
    print(f"      Added from REVIEW: {len(review_keep_df)}")
    print(f"    CONTEXT (saved separately): {len(review_context_df)}")
    print(f"    DROPPED: {len(review_drop_df)}")
    
    # Save outputs
    base_name = os.path.basename(main_file).replace('.csv', '')
    os.makedirs(output_dir, exist_ok=True)
    
    # Update main file with KEEP + REVIEW_KEEP
    main_output = os.path.join(output_dir, f"{base_name}.csv")
    final_main_df.to_csv(main_output, index=False, encoding='utf-8-sig')
    print(f"  ✓ Updated main file: {main_output}")
    
    # Save CONTEXT samples separately
    if len(review_context_df) > 0:
        context_output = os.path.join(output_dir, f"{base_name}_CONTEXT.csv")
        review_context_df.to_csv(context_output, index=False, encoding='utf-8-sig')
        print(f"  ✓ Saved CONTEXT: {context_output}")
    
    # Save DROPPED samples for audit
    if len(review_drop_df) > 0:
        dropped_output = os.path.join(output_dir, f"{base_name}_DROPPED_FROM_REVIEW.csv")
        review_drop_df.to_csv(dropped_output, index=False, encoding='utf-8-sig')
        print(f"  ✓ Saved DROPPED: {dropped_output}")
    
    # Statistics
    stats = {
        'review_file': review_file,
        'main_file': main_file,
        'original_keep': keep_count,
        'original_review': review_count,
        'review_keep_added': len(review_keep_df),
        'review_context': len(review_context_df),
        'review_dropped': len(review_drop_df),
        'final_main_count': len(final_main_df),
        'addition_rate': len(review_keep_df) / review_count if review_count > 0 else 0
    }
    
    return stats

def batch_process():
    """Batch process all REVIEW files and merge back to main files"""
    input_dir = '01_Data/03_filtered_datasets'
    output_dir = '01_Data/04_refined_datasets'
    
    datasets = ['01_news_datasets', '02_conference_datasets', '03_X_datasets']
    
    # Find all *_REVIEW.csv files
    review_files = []
    for dataset in datasets:
        dataset_path = os.path.join(input_dir, dataset)
        if os.path.exists(dataset_path):
            files = glob.glob(os.path.join(dataset_path, '*_REVIEW.csv'))
            review_files.extend([(f, dataset) for f in files])
    
    print(f"Found {len(review_files)} REVIEW files to process")
    print("=" * 60)
    
    all_stats = []
    
    for review_file, dataset in review_files:
        # Find corresponding main file (remove _REVIEW.csv, add .csv)
        base_name = os.path.basename(review_file).replace('_REVIEW.csv', '.csv')
        main_file = os.path.join(os.path.dirname(review_file), base_name)
        output_dir_path = os.path.join(output_dir, dataset)
        
        try:
            stats = process_review_file(review_file, main_file, output_dir_path)
            all_stats.append(stats)
        except Exception as e:
            print(f"  ✗ Error processing {os.path.basename(review_file)}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save overall statistics
    stats_output = os.path.join(output_dir, '_post_filtering_stats.json')
    os.makedirs(os.path.dirname(stats_output), exist_ok=True)
    
    with open(stats_output, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("BATCH POST-FILTERING COMPLETED!")
    print("=" * 60)
    
    total_keep = sum(s['original_keep'] for s in all_stats)
    total_review = sum(s['original_review'] for s in all_stats)
    total_added = sum(s['review_keep_added'] for s in all_stats)
    total_context = sum(s['review_context'] for s in all_stats)
    total_dropped = sum(s['review_dropped'] for s in all_stats)
    total_final = sum(s['final_main_count'] for s in all_stats)
    
    print(f"\nOverall statistics:")
    print(f"  Original KEEP samples: {total_keep}")
    print(f"  Original REVIEW samples: {total_review}")
    print(f"\n  REVIEW processing:")
    print(f"    Added to main (REVIEW_KEEP): {total_added} ({total_added/total_review*100:.1f}%)")
    print(f"    Saved as CONTEXT: {total_context} ({total_context/total_review*100:.1f}%)")
    print(f"    Dropped: {total_dropped} ({total_dropped/total_review*100:.1f}%)")
    print(f"\n  Final main files: {total_final} samples")
    print(f"    (Original KEEP + Added from REVIEW)")
    print(f"\nStatistics saved to: {stats_output}")
    print(f"Refined datasets saved to: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    batch_process()

