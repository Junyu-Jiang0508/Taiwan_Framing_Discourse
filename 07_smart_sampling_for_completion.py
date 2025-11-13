"""
Smart sampling for annotation completion
Based on L1/L2 gap analysis and diversity constraints
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import random

class SmartSampler:
    def __init__(self, statistics_path: str, target_tier: str = 'A'):
        """
        Args:
            statistics_path: Path to annotation statistics JSON
            target_tier: 'A' (baseline, +150) or 'B' (recommended, +350)
        """
        with open(statistics_path, 'r', encoding='utf-8') as f:
            self.stats = json.load(f)
        
        self.target_tier = target_tier
        self.l1_targets = {'A': 25, 'B': 50, 'C': 100}
        self.l2_targets = {'A': 30, 'B': 50, 'C': 75}
        
        self.l1_gaps = self._calculate_l1_gaps()
        self.l2_gaps = self._calculate_l2_gaps()
        
    def _calculate_l1_gaps(self) -> Dict[str, int]:
        """Calculate how many samples needed for each L1 label"""
        target = self.l1_targets[self.target_tier]
        gaps = {}
        
        for label, count in self.stats['l1_distribution'].items():
            if count < target:
                gaps[label] = target - count
        
        return gaps
    
    def _calculate_l2_gaps(self) -> Dict[str, int]:
        """Calculate how many samples needed for each L2 label"""
        target = self.l2_targets[self.target_tier]
        gaps = {}
        
        for label, count in self.stats['l2_distribution'].items():
            if count < target:
                gaps[label] = target - count
        
        return gaps
    
    def print_gaps(self):
        """Print current gaps"""
        print("=" * 80)
        print(f"SAMPLING PLAN - TIER {self.target_tier}")
        print("=" * 80)
        
        print(f"\nL1 Target: {self.l1_targets[self.target_tier]} per label")
        print(f"L1 Gaps:")
        total_l1_gap = sum(self.l1_gaps.values())
        for label, gap in sorted(self.l1_gaps.items(), key=lambda x: x[1], reverse=True):
            print(f"  {label}: need +{gap}")
        print(f"  Total L1 gap: +{total_l1_gap} samples")
        
        print(f"\nL2 Target: {self.l2_targets[self.target_tier]} per label")
        print(f"L2 Gaps:")
        total_l2_gap = sum(self.l2_gaps.values())
        for label, gap in sorted(self.l2_gaps.items(), key=lambda x: x[1], reverse=True):
            print(f"  {label}: need +{gap} instances")
        print(f"  Total L2 gap: +{total_l2_gap} instances")
        
        avg_l2_per_sample = self.stats.get('l2_avg_per_sample', 1.74)
        estimated_for_l2 = int(total_l2_gap / avg_l2_per_sample)
        
        print(f"\nEstimated samples needed:")
        print(f"  By L1 gaps: +{total_l1_gap}")
        print(f"  By L2 gaps: +{estimated_for_l2} (assuming {avg_l2_per_sample:.2f} L2/sample)")
        print(f"  Recommendation: +{max(total_l1_gap, estimated_for_l2)} samples")
        
        return max(total_l1_gap, estimated_for_l2)

    def load_refined_datasets(self) -> pd.DataFrame:
        """Load all refined datasets for sampling"""
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
                    df = pd.read_csv(csv_file, encoding='utf-8-sig')
                    df['source_file'] = csv_file.stem
                    all_data.append(df)
                    print(f"Loaded: {csv_file.name} ({len(df)} samples)")
                except Exception as e:
                    print(f"Error loading {csv_file.name}: {e}")
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            print(f"\nTotal refined samples available: {len(combined)}")
            return combined
        
        return pd.DataFrame()
    
    def score_sample_for_gaps(self, row: pd.Series, l2_keywords: Dict = None) -> Tuple[float, List[str]]:
        """
        Score a sample based on how well it fills gaps
        Returns: (score, reason_list)
        """
        score = 0.0
        reasons = []
        
        text = str(row.get('sentence', row.get('text', '')))
        
        # Check if sample is in boundary zone (higher priority)
        if 'semantic_similarity' in row:
            sim = row['semantic_similarity']
            if pd.notna(sim) and 0.45 <= sim <= 0.60:
                score += 2.0
                reasons.append('boundary_semantic')
        
        if 'nli_score' in row:
            nli = row['nli_score']
            if pd.notna(nli) and 0.05 <= nli <= 0.20:
                score += 1.5
                reasons.append('boundary_nli')
        
        # Bonus for hitting scarce L2 keywords
        if l2_keywords:
            l2_hits = 0
            for l2_label, keywords in l2_keywords.items():
                if l2_label in self.l2_gaps:  # Only check gaps
                    if any(kw in text for kw in keywords):
                        l2_hits += 1
                        reasons.append(f'l2_{l2_label}')
            
            # Bonus for each scarce L2 hit
            score += l2_hits * 1.0
        
        # Check for speakers (higher priority)
        if 'speakers' in row and row['speakers']:
            score += 1.0
            reasons.append('has_speaker')
        
        # Check for targets
        if 'targets' in row and row['targets']:
            score += 0.8
            reasons.append('has_target')
        
        # Check for procedural/symbolic markers
        if 'proc_symbolic_flag' in row and row['proc_symbolic_flag']:
            score += 0.5
            reasons.append('procedural')
        
        # Prefer quote and claim over context
        if 'role' in row:
            if row['role'] == 'quote':
                score += 1.5
                reasons.append('quote')
            elif row['role'] == 'claim':
                score += 1.0
                reasons.append('claim')
        
        # Length preference (not too short, not too long)
        text_len = len(text)
        if 50 <= text_len <= 200:
            score += 0.5
            reasons.append('good_length')
        
        return score, reasons
    
    def sample_for_l1_gaps(self, df: pd.DataFrame, quota_per_label: Dict[str, int]) -> pd.DataFrame:
        """
        Sample candidates to fill L1 gaps with L2 consideration
        Uses keyword matching + diversity constraints + L2 gap prioritization
        """
        # L1 label keywords
        l1_keywords = {
            'L1-01': ['經濟', '经济', '發展', '发展', '投資', '投资', '貿易', '贸易', '就業', '就业', '薪資', '薪资'],
            'L1-02': ['道德', '倫理', '伦理', '價值', '价值', '正義', '正义', '品德', '操守'],
            'L1-03': ['責任', '责任', '問題', '问题', '批評', '批评', '追究', '究責', '究责'],
            'L1-04': ['衝突', '冲突', '對立', '对立', '矛盾', '爭議', '争议', '對抗', '对抗'],
            'L1-05': ['人性', '同情', '關懷', '关怀', '感情', '同理', '溫暖', '温暖'],
            'L1-06': ['公平', '平等', '權利', '权利', '保障', '權益', '权益'],
            'L1-07': ['法律', '法規', '法规', '制度', '依法', '合法', '違法', '违法', '憲法', '宪法'],
            'L1-08': ['政策', '方案', '計畫', '计划', '措施', '推動', '推动', '執行', '执行'],
            'L1-09': ['主權', '主权', '認同', '认同', '兩岸', '两岸', '統獨', '统独', '國家', '国家'],
            'L1-14': ['領導', '领导', '能力', '經驗', '经验', '資格', '资格', '執政', '执政'],
            'L1-15': ['安全', '風險', '风险', '威脅', '威胁', '危機', '危机', '國防', '国防'],
        }
        
        # L2 label keywords (for gap filling)
        l2_keywords = {
            'L2-01': ['主權', '主权', '獨立', '独立', '國家', '国家'],
            'L2-02': ['民主', '自由', '人權', '人权'],
            'L2-03': ['經濟', '经济', '繁榮', '繁荣', '發展', '发展'],
            'L2-04': ['兩岸', '两岸', '和平', '穩定', '稳定'],
            'L2-05': ['社會', '社会', '正義', '正义', '公平', '福利'],
            'L2-06': ['文化', '傳統', '传统', '歷史', '历史'],
            'L2-07': ['安全', '國防', '国防', '威脅', '威胁'],
            'L2-08': ['認同', '认同', '本土', '臺灣', '台湾'],
            'L2-09': ['制度', '憲政', '宪政', '法治'],
            'L2-10': ['外交', '國際', '国际', '地位'],
            'L2-11': ['環境', '环境', '永續', '永续', '生態', '生态'],
            'L2-12': ['教育', '文化', '傳承', '传承'],
            'L2-13': ['創新', '创新', '科技', '轉型', '转型'],
            'L2-14': ['團結', '团结', '凝聚', '共識', '共识'],
            'L2-15': ['改革', '變革', '变革', '進步', '进步'],
        }
        
        sampled = []
        
        for label, quota in quota_per_label.items():
            print(f"\nSampling for {label}: need {quota} samples")
            
            # Get candidates with keyword matching
            keywords = l1_keywords.get(label, [])
            if keywords:
                mask = df['sentence'].str.contains('|'.join(keywords), case=False, na=False)
                candidates = df[mask].copy()
                
                # If not enough keyword matches, add random samples
                if len(candidates) < quota * 3:
                    additional_needed = quota * 3 - len(candidates)
                    remaining_df = df[~mask]
                    if len(remaining_df) > 0:
                        additional = remaining_df.sample(n=min(additional_needed, len(remaining_df)))
                        candidates = pd.concat([candidates, additional], ignore_index=True)
            else:
                # If no keywords, sample randomly with larger pool
                candidates = df.sample(n=min(quota * 5, len(df)))
            
            if len(candidates) == 0:
                print(f"  Warning: No candidates found for {label}")
                continue
            
            # Score each candidate (with L2 gap consideration)
            candidates['sample_score'] = candidates.apply(
                lambda row: self.score_sample_for_gaps(row, l2_keywords)[0], axis=1
            )
            candidates['score_reasons'] = candidates.apply(
                lambda row: '|'.join(self.score_sample_for_gaps(row, l2_keywords)[1]), axis=1
            )
            
            # Sort by score and take top candidates
            candidates = candidates.sort_values('sample_score', ascending=False)
            
            # Apply diversity constraints
            # Adjust oversampling ratio based on quota size
            if quota <= 20:
                oversample_ratio = 2.0
            elif quota <= 50:
                oversample_ratio = 1.5
            else:  # Large gaps
                oversample_ratio = 1.3
            
            target_count = int(quota * oversample_ratio)
            
            selected = []
            used_sources = []
            used_dates = []
            
            for idx, row in candidates.iterrows():
                if len(selected) >= target_count:
                    break
                
                # Diversity check (relaxed for large quotas)
                source = row.get('source_file', '')
                date = row.get('date', '')
                
                # Allow more repetition for large gaps
                if quota > 80:
                    max_source_repeat = 10
                    max_date_repeat = 5
                elif quota > 50:
                    max_source_repeat = 7
                    max_date_repeat = 4
                else:
                    max_source_repeat = 3
                    max_date_repeat = 2
                
                if source and used_sources.count(source) >= max_source_repeat:
                    continue
                if date and used_dates.count(date) >= max_date_repeat:
                    continue
                
                selected.append(row)
                used_sources.append(source)
                used_dates.append(date)
            
            selected_df = pd.DataFrame(selected)
            selected_df['target_l1'] = label
            selected_df['sampling_reason'] = 'l1_gap_filling'
            sampled.append(selected_df)
            
            ratio = len(selected) / quota if quota > 0 else 0
            print(f"  Selected {len(selected)} candidates ({ratio:.1f}x quota, target={target_count})")
        
        if sampled:
            return pd.concat(sampled, ignore_index=True)
        return pd.DataFrame()
    
    def export_sampling_candidates(self, candidates: pd.DataFrame, output_dir: str):
        """Export candidates for manual annotation"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Main candidate list
        main_output = output_path / f'candidates_tier_{self.target_tier}.csv'
        
        # Select key columns for annotation
        output_cols = [
            'unit_id', 'sentence', 'role', 'prev', 'next',
            'speakers', 'targets', 'proc_symbolic_flag', 'event_phase',
            'source', 'date', 'target_l1', 'sample_score', 'score_reasons', 'sampling_reason'
        ]
        
        available_cols = [col for col in output_cols if col in candidates.columns]
        candidates[available_cols].to_csv(main_output, index=False, encoding='utf-8-sig')
        
        print(f"\n[OK] Main candidate list saved: {main_output}")
        
        # Split by target L1 for easier annotation
        for label in candidates['target_l1'].unique():
            label_df = candidates[candidates['target_l1'] == label]
            label_output = output_path / f'candidates_{label}_tier_{self.target_tier}.csv'
            label_df[available_cols].to_csv(label_output, index=False, encoding='utf-8-sig')
            print(f"  [OK] {label}: {len(label_df)} candidates")
        
        # Analyze L2 coverage in candidates
        l2_coverage = {}
        for l2_label in self.l2_gaps.keys():
            hit_count = candidates['score_reasons'].str.contains(f'l2_{l2_label}', na=False).sum()
            l2_coverage[l2_label] = int(hit_count)
        
        print(f"\n{'='*80}")
        print("L2 GAP COVERAGE PREDICTION")
        print(f"{'='*80}")
        print("\nScarce L2 labels potentially covered by candidates:")
        for l2_label, gap in sorted(self.l2_gaps.items(), key=lambda x: x[1], reverse=True):
            covered = l2_coverage.get(l2_label, 0)
            print(f"  {l2_label}: gap={gap}, candidates_with_keywords={covered}")
        
        # Summary statistics
        summary = {
            'tier': self.target_tier,
            'total_candidates': len(candidates),
            'l1_gaps': self.l1_gaps,
            'l2_gaps': self.l2_gaps,
            'l2_coverage_prediction': l2_coverage,
            'candidates_by_l1': candidates['target_l1'].value_counts().to_dict(),
            'avg_sample_score': float(candidates['sample_score'].mean()),
            'role_distribution': candidates['role'].value_counts().to_dict() if 'role' in candidates.columns else {},
            'source_diversity': candidates['source'].nunique() if 'source' in candidates.columns else 0,
        }
        
        summary_output = output_path / f'sampling_summary_tier_{self.target_tier}.json'
        with open(summary_output, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n[OK] Sampling summary saved: {summary_output}")

def main():
    # Load statistics
    stats_path = '01_Data/06_manual_sets/03_results/03_deepseek_annotations_statistics.json'
    
    # Choose tier: 'A' for +150, 'B' for +350, 'C' for +700-900
    tier = 'C'  # C for research publication / audit-friendly
    
    print("=" * 80)
    print("SMART SAMPLING FOR ANNOTATION COMPLETION")
    print("=" * 80)
    
    sampler = SmartSampler(stats_path, target_tier=tier)
    
    # Print gap analysis
    estimated_needed = sampler.print_gaps()
    
    print("\n" + "=" * 80)
    print("LOADING REFINED DATASETS")
    print("=" * 80)
    
    # Load available data
    df = sampler.load_refined_datasets()
    
    if len(df) == 0:
        print("Error: No data available for sampling")
        return
    
    print("\n" + "=" * 80)
    print("SAMPLING CANDIDATES")
    print("=" * 80)
    
    # Sample for L1 gaps
    candidates = sampler.sample_for_l1_gaps(df, sampler.l1_gaps)
    
    if len(candidates) == 0:
        print("Error: No candidates sampled")
        return
    
    print(f"\nTotal candidates sampled: {len(candidates)}")
    print(f"Average sample score: {candidates['sample_score'].mean():.2f}")
    
    total_l1_gap = sum(sampler.l1_gaps.values())
    coverage_rate = len(candidates) / total_l1_gap if total_l1_gap > 0 else 0
    print(f"\nGap coverage:")
    print(f"  Total L1 gap: +{total_l1_gap}")
    print(f"  Candidates generated: {len(candidates)}")
    print(f"  Coverage rate: {coverage_rate:.1%}")
    
    if coverage_rate < 0.9:
        print(f"\n  Note: Coverage < 90%. Consider:")
        print(f"    - Relaxing diversity constraints further")
        print(f"    - Expanding keyword lists")
        print(f"    - Using semantic clustering for hard-to-find labels")
    
    # Export
    output_dir = '01_Data/06_manual_sets/02_sampling_candidates'
    sampler.export_sampling_candidates(candidates, output_dir)
    
    print("\n" + "=" * 80)
    print("SAMPLING COMPLETE")
    print("=" * 80)
    print(f"\nNext steps:")
    print(f"1. Review candidates in: {output_dir}")
    print(f"2. Manually annotate selected samples")
    print(f"3. Merge with existing annotations")
    print(f"4. Re-run statistics to verify gaps are filled")

if __name__ == "__main__":
    main()

