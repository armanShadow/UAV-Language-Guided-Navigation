#!/usr/bin/env python3
"""
Paraphrase Verification Script
=============================

Verifies that each dialog turn with an answer in the augmented AVDN dataset has:
1. Exactly 2 positive paraphrases and 1 negative paraphrase
2. Proper validation results for each paraphrase
3. Expected validation criteria (valid positives, valid negatives)

FEATURES:
✅ Comprehensive structure validation
✅ Validation result analysis
✅ Success/failure statistics per split
✅ Detailed reports on issues found
✅ Validation threshold analysis
✅ Quality metrics summary

USAGE:
    python verify_paraphrases.py
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ParaphraseVerifier:
    """
    Comprehensive verification of augmented AVDN dataset paraphrases.
    Checks structure, validation results, and quality metrics.
    """
    
    def __init__(self):
        # Augmented dataset paths
        self.dataset_paths = {
            'train': "augmented_data/train_data_with_paraphrases.json",
            'val_seen': "augmented_data/val_seen_data_with_paraphrases.json", 
            'val_unseen': "augmented_data/val_unseen_data_with_paraphrases.json"
        }
        
        # Statistics tracking
        self.stats = {
            'train': self._init_split_stats(),
            'val_seen': self._init_split_stats(),
            'val_unseen': self._init_split_stats()
        }
        
        # Issues tracking
        self.issues = {
            'train': defaultdict(list),
            'val_seen': defaultdict(list),
            'val_unseen': defaultdict(list)
        }
        
        logger.info("🔍 Paraphrase Verifier initialized")
        logger.info("📂 Will verify all augmented dataset splits")
    
    def _init_split_stats(self) -> Dict:
        """Initialize statistics structure for a split."""
        return {
            'total_episodes': 0,
            'total_dialogs': 0,
            'dialogs_with_answers': 0,
            'dialogs_with_paraphrases': 0,
            'correct_structure': 0,  # Exactly 2P + 1N
            'validation_issues': 0,
            'perfect_validations': 0,  # All paraphrases valid
            'partial_validations': 0,  # Some paraphrases valid
            'failed_validations': 0,   # No paraphrases valid
            'total_positives_generated': 0,
            'total_negatives_generated': 0,
            'total_valid_positives': 0,
            'total_valid_negatives': 0
        }
    
    def verify_all_splits(self) -> Dict:
        """Verify all dataset splits and return comprehensive results."""
        logger.info("🚀 Starting comprehensive paraphrase verification...")
        
        results = {}
        
        for split in ['train', 'val_seen', 'val_unseen']:
            logger.info(f"\n{'='*60}")
            logger.info(f"🔍 Verifying {split.upper()} split")
            logger.info(f"{'='*60}")
            
            results[split] = self.verify_split(split)
        
        # Generate final summary
        self._log_comprehensive_summary(results)
        
        return results
    
    def verify_split(self, split: str) -> Dict:
        """Verify a specific dataset split."""
        try:
            # Load augmented dataset
            dataset_path = self.dataset_paths[split]
            
            if not Path(dataset_path).exists():
                logger.error(f"❌ Augmented dataset not found: {dataset_path}")
                return {'success': False, 'error': f'File not found: {dataset_path}'}
            
            logger.info(f"📂 Loading {split} augmented dataset...")
            with open(dataset_path, 'r') as f:
                episodes = json.load(f)
            
            logger.info(f"📊 Loaded {len(episodes)} episodes")
            
            # Verify each episode
            for episode in episodes:
                self._verify_episode(episode, split)
            
            # Calculate summary statistics
            split_stats = self.stats[split]
            self._calculate_summary_stats(split_stats)
            
            # Log split results
            self._log_split_results(split)
            
            return {
                'success': True,
                'statistics': split_stats,
                'issues': dict(self.issues[split])
            }
            
        except Exception as e:
            logger.error(f"❌ Error verifying {split} split: {e}")
            return {'success': False, 'error': str(e)}
    
    def _verify_episode(self, episode: Dict, split: str):
        """Verify a single episode."""
        episode_id = episode.get('episode_id', 'unknown')
        dialogs = episode.get('dialogs', [])
        
        self.stats[split]['total_episodes'] += 1
        self.stats[split]['total_dialogs'] += len(dialogs)
        
        for turn_idx, dialog in enumerate(dialogs):
            # Check if dialog has an answer
            answer = dialog.get('answer')
            
            if answer and answer.strip() and len(answer.strip().split()) >= 3:
                self.stats[split]['dialogs_with_answers'] += 1
                
                # Check if paraphrases exist
                paraphrases = dialog.get('paraphrases')
                
                if paraphrases:
                    self.stats[split]['dialogs_with_paraphrases'] += 1
                    self._verify_paraphrase_structure(episode_id, turn_idx, paraphrases, split)
                else:
                    # Missing paraphrases
                    issue = f"Episode {episode_id}, Turn {turn_idx}: Missing paraphrases field for answer '{answer[:50]}...'"
                    self.issues[split]['missing_paraphrases'].append(issue)
    
    def _verify_paraphrase_structure(self, episode_id: str, turn_idx: int, paraphrases: Dict, split: str):
        """Verify the structure and validation of paraphrases."""
        context = f"Episode {episode_id}, Turn {turn_idx}"
        
        # Extract paraphrase data
        positives = paraphrases.get('positives', [])
        negatives = paraphrases.get('negatives', [])
        valid_positives = paraphrases.get('valid_positives', [])
        valid_negatives = paraphrases.get('valid_negatives', [])
        validation_analysis = paraphrases.get('validation_analysis', {})
        
        # Update generation counts
        self.stats[split]['total_positives_generated'] += len(positives)
        self.stats[split]['total_negatives_generated'] += len(negatives)
        self.stats[split]['total_valid_positives'] += len(valid_positives)
        self.stats[split]['total_valid_negatives'] += len(valid_negatives)
        
        # Check structure: Should have exactly 2 positives and 1 negative
        expected_structure = len(positives) == 2 and len(negatives) == 1
        
        if expected_structure:
            self.stats[split]['correct_structure'] += 1
        else:
            issue = f"{context}: Incorrect structure - {len(positives)} positives, {len(negatives)} negatives (expected 2P + 1N)"
            self.issues[split]['structure_issues'].append(issue)
        
        # Check validation results
        validation_success = self._analyze_validation_results(
            context, positives, negatives, valid_positives, valid_negatives, 
            validation_analysis, split
        )
        
        # Categorize validation success
        total_valid = len(valid_positives) + len(valid_negatives)
        total_generated = len(positives) + len(negatives)
        
        if total_valid == total_generated and total_valid > 0:
            self.stats[split]['perfect_validations'] += 1
        elif total_valid > 0:
            self.stats[split]['partial_validations'] += 1
        else:
            self.stats[split]['failed_validations'] += 1
    
    def _analyze_validation_results(self, context: str, positives: List[str], negatives: List[str], 
                                  valid_positives: List[str], valid_negatives: List[str], 
                                  validation_analysis: Dict, split: str) -> bool:
        """Analyze validation results in detail."""
        
        # Check if validation analysis exists
        if not validation_analysis:
            issue = f"{context}: Missing validation_analysis field"
            self.issues[split]['validation_issues'].append(issue)
            self.stats[split]['validation_issues'] += 1
            return False
        
        # Check validation details structure
        validation_details = validation_analysis.get('validation_details', {})
        positive_results = validation_details.get('positive_results', [])
        negative_results = validation_details.get('negative_results', [])
        
        # Verify positive validations
        for i, positive in enumerate(positives):
            if i < len(positive_results):
                pos_result = positive_results[i]
                is_valid = pos_result.get('is_valid', False)
                embedding_sim = pos_result.get('embedding_similarity', 0.0)
                
                if positive in valid_positives and not is_valid:
                    issue = f"{context}: Positive '{positive[:30]}...' marked valid but validation says invalid"
                    self.issues[split]['validation_inconsistencies'].append(issue)
                elif positive not in valid_positives and is_valid:
                    issue = f"{context}: Positive '{positive[:30]}...' marked invalid but validation says valid"
                    self.issues[split]['validation_inconsistencies'].append(issue)
                
                # Check embedding similarity thresholds
                if not is_valid and embedding_sim < 0.5:
                    issue = f"{context}: Positive rejected - low embedding similarity ({embedding_sim:.3f})"
                    self.issues[split]['low_embedding_similarity'].append(issue)
            else:
                issue = f"{context}: Missing validation result for positive {i}"
                self.issues[split]['missing_validation_results'].append(issue)
        
        # Verify negative validations
        for i, negative in enumerate(negatives):
            if i < len(negative_results):
                neg_result = negative_results[i]
                is_valid = neg_result.get('is_valid', False)
                embedding_sim = neg_result.get('embedding_similarity', 0.0)
                spatial_changed = neg_result.get('spatial_changed', False)
                
                if negative in valid_negatives and not is_valid:
                    issue = f"{context}: Negative '{negative[:30]}...' marked valid but validation says invalid"
                    self.issues[split]['validation_inconsistencies'].append(issue)
                elif negative not in valid_negatives and is_valid:
                    issue = f"{context}: Negative '{negative[:30]}...' marked invalid but validation says valid"
                    self.issues[split]['validation_inconsistencies'].append(issue)
                
                # Check negative validation criteria
                if not is_valid:
                    if embedding_sim >= 0.92:
                        issue = f"{context}: Negative rejected - too similar ({embedding_sim:.3f})"
                        self.issues[split]['high_embedding_similarity'].append(issue)
                    elif not spatial_changed:
                        issue = f"{context}: Negative rejected - insufficient spatial changes"
                        self.issues[split]['insufficient_spatial_changes'].append(issue)
            else:
                issue = f"{context}: Missing validation result for negative {i}"
                self.issues[split]['missing_validation_results'].append(issue)
        
        return True
    
    def _calculate_summary_stats(self, split_stats: Dict):
        """Calculate derived statistics."""
        # Paraphrase coverage
        if split_stats['dialogs_with_answers'] > 0:
            split_stats['paraphrase_coverage'] = split_stats['dialogs_with_paraphrases'] / split_stats['dialogs_with_answers']
        else:
            split_stats['paraphrase_coverage'] = 0.0
        
        # Structure correctness
        if split_stats['dialogs_with_paraphrases'] > 0:
            split_stats['structure_correctness'] = split_stats['correct_structure'] / split_stats['dialogs_with_paraphrases']
        else:
            split_stats['structure_correctness'] = 0.0
        
        # Validation success rates
        if split_stats['total_positives_generated'] > 0:
            split_stats['positive_validation_rate'] = split_stats['total_valid_positives'] / split_stats['total_positives_generated']
        else:
            split_stats['positive_validation_rate'] = 0.0
        
        if split_stats['total_negatives_generated'] > 0:
            split_stats['negative_validation_rate'] = split_stats['total_valid_negatives'] / split_stats['total_negatives_generated']
        else:
            split_stats['negative_validation_rate'] = 0.0
        
        # Overall validation rate
        total_generated = split_stats['total_positives_generated'] + split_stats['total_negatives_generated']
        total_valid = split_stats['total_valid_positives'] + split_stats['total_valid_negatives']
        
        if total_generated > 0:
            split_stats['overall_validation_rate'] = total_valid / total_generated
        else:
            split_stats['overall_validation_rate'] = 0.0
    
    def _log_split_results(self, split: str):
        """Log detailed results for a split."""
        stats = self.stats[split]
        issues = self.issues[split]
        
        logger.info(f"\n📊 {split.upper()} VERIFICATION RESULTS:")
        logger.info(f"{'='*50}")
        
        # Basic counts
        logger.info(f"📂 Dataset Structure:")
        logger.info(f"  Total episodes: {stats['total_episodes']}")
        logger.info(f"  Total dialog turns: {stats['total_dialogs']}")
        logger.info(f"  Dialog turns with answers: {stats['dialogs_with_answers']}")
        logger.info(f"  Dialog turns with paraphrases: {stats['dialogs_with_paraphrases']}")
        logger.info(f"  Paraphrase coverage: {stats['paraphrase_coverage']:.2%}")
        
        # Structure verification
        logger.info(f"\n🏗️ Structure Verification:")
        logger.info(f"  Correct structure (2P + 1N): {stats['correct_structure']}")
        logger.info(f"  Structure correctness: {stats['structure_correctness']:.2%}")
        
        # Generation counts
        logger.info(f"\n📝 Generation Results:")
        logger.info(f"  Total positives generated: {stats['total_positives_generated']}")
        logger.info(f"  Total negatives generated: {stats['total_negatives_generated']}")
        logger.info(f"  Expected positives: {stats['dialogs_with_paraphrases'] * 2}")
        logger.info(f"  Expected negatives: {stats['dialogs_with_paraphrases'] * 1}")
        
        # Validation results
        logger.info(f"\n✅ Validation Results:")
        logger.info(f"  Valid positives: {stats['total_valid_positives']}")
        logger.info(f"  Valid negatives: {stats['total_valid_negatives']}")
        logger.info(f"  Positive validation rate: {stats['positive_validation_rate']:.2%}")
        logger.info(f"  Negative validation rate: {stats['negative_validation_rate']:.2%}")
        logger.info(f"  Overall validation rate: {stats['overall_validation_rate']:.2%}")
        
        # Validation quality distribution
        logger.info(f"\n🎯 Validation Quality:")
        logger.info(f"  Perfect validations (all valid): {stats['perfect_validations']}")
        logger.info(f"  Partial validations (some valid): {stats['partial_validations']}")
        logger.info(f"  Failed validations (none valid): {stats['failed_validations']}")
        
        # Issues summary
        total_issues = sum(len(issue_list) for issue_list in issues.values())
        logger.info(f"\n⚠️ Issues Found: {total_issues}")
        
        for issue_type, issue_list in issues.items():
            if issue_list:
                logger.info(f"  {issue_type}: {len(issue_list)}")
                # Show first few examples
                for i, issue in enumerate(issue_list[:3]):
                    logger.info(f"    - {issue}")
                if len(issue_list) > 3:
                    logger.info(f"    ... and {len(issue_list) - 3} more")
    
    def _log_comprehensive_summary(self, results: Dict):
        """Log comprehensive summary across all splits."""
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 COMPREHENSIVE VERIFICATION SUMMARY")
        logger.info(f"{'='*60}")
        
        # Aggregate statistics
        total_episodes = sum(self.stats[split]['total_episodes'] for split in self.stats)
        total_dialogs_with_answers = sum(self.stats[split]['dialogs_with_answers'] for split in self.stats)
        total_dialogs_with_paraphrases = sum(self.stats[split]['dialogs_with_paraphrases'] for split in self.stats)
        total_correct_structure = sum(self.stats[split]['correct_structure'] for split in self.stats)
        total_positives_generated = sum(self.stats[split]['total_positives_generated'] for split in self.stats)
        total_negatives_generated = sum(self.stats[split]['total_negatives_generated'] for split in self.stats)
        total_valid_positives = sum(self.stats[split]['total_valid_positives'] for split in self.stats)
        total_valid_negatives = sum(self.stats[split]['total_valid_negatives'] for split in self.stats)
        
        logger.info(f"🎯 OVERALL TOTALS:")
        logger.info(f"  Total episodes: {total_episodes}")
        logger.info(f"  Dialog turns with answers: {total_dialogs_with_answers}")
        logger.info(f"  Dialog turns with paraphrases: {total_dialogs_with_paraphrases}")
        logger.info(f"  Paraphrase coverage: {total_dialogs_with_paraphrases/max(1,total_dialogs_with_answers):.2%}")
        
        logger.info(f"\n🏗️ STRUCTURE VERIFICATION:")
        logger.info(f"  Correct structure (2P + 1N): {total_correct_structure}")
        logger.info(f"  Structure correctness: {total_correct_structure/max(1,total_dialogs_with_paraphrases):.2%}")
        
        logger.info(f"\n📊 GENERATION TOTALS:")
        logger.info(f"  Positives generated: {total_positives_generated} (expected: {total_dialogs_with_paraphrases * 2})")
        logger.info(f"  Negatives generated: {total_negatives_generated} (expected: {total_dialogs_with_paraphrases * 1})")
        
        logger.info(f"\n✅ VALIDATION TOTALS:")
        logger.info(f"  Valid positives: {total_valid_positives}/{total_positives_generated} ({total_valid_positives/max(1,total_positives_generated):.2%})")
        logger.info(f"  Valid negatives: {total_valid_negatives}/{total_negatives_generated} ({total_valid_negatives/max(1,total_negatives_generated):.2%})")
        logger.info(f"  Overall validation rate: {(total_valid_positives+total_valid_negatives)/max(1,total_positives_generated+total_negatives_generated):.2%}")
        
        # Success assessment
        success_criteria = [
            total_dialogs_with_paraphrases == total_dialogs_with_answers,  # 100% coverage
            total_correct_structure == total_dialogs_with_paraphrases,    # 100% correct structure
            total_positives_generated == total_dialogs_with_paraphrases * 2,  # Correct positive count
            total_negatives_generated == total_dialogs_with_paraphrases * 1,  # Correct negative count
        ]
        
        passed_criteria = sum(success_criteria)
        logger.info(f"\n🎯 SUCCESS CRITERIA:")
        logger.info(f"  Criteria passed: {passed_criteria}/4")
        logger.info(f"  ✅ 100% paraphrase coverage: {'YES' if success_criteria[0] else 'NO'}")
        logger.info(f"  ✅ 100% correct structure: {'YES' if success_criteria[1] else 'NO'}")
        logger.info(f"  ✅ Correct positive count: {'YES' if success_criteria[2] else 'NO'}")
        logger.info(f"  ✅ Correct negative count: {'YES' if success_criteria[3] else 'NO'}")
        
        if passed_criteria == 4:
            logger.info(f"🎉 VERIFICATION PASSED: All criteria met!")
        else:
            logger.warning(f"⚠️ VERIFICATION ISSUES: {4-passed_criteria} criteria failed")

def main():
    """Run the paraphrase verification."""
    verifier = ParaphraseVerifier()
    
    try:
        # Verify all splits
        results = verifier.verify_all_splits()
        
        # Check overall success
        all_successful = all(result.get('success', False) for result in results.values())
        
        if all_successful:
            logger.info("✅ Verification completed successfully for all splits")
        else:
            failed_splits = [split for split, result in results.items() if not result.get('success', False)]
            logger.error(f"❌ Verification failed for splits: {failed_splits}")
        
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")

if __name__ == "__main__":
    main() 