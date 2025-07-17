#!/usr/bin/env python3
"""
Structure Issues Fix Script
==========================

Regenerates paraphrases ONLY for dialog turns that have structure issues 
(not exactly 2 positives + 1 negative). This is a targeted fix script.

FEATURES:
✅ Identifies dialog turns with structure issues
✅ Regenerates paraphrases for those specific turns only
✅ Preserves all other correctly structured dialog turns
✅ Maintains original dataset structure
✅ Comprehensive logging and validation

USAGE:
    python fix_structure_issues.py
"""

import json
import logging
import torch
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

# Import pipeline components
from paraphrase_generation_pipeline import ParaphraseGenerationPipeline
from validation_pipeline import ValidationPipeline

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StructureIssueFixer:
    """
    Targeted fixer for structure issues in augmented AVDN dataset.
    Only regenerates paraphrases for dialog turns with incorrect structure.
    """
    
    def __init__(self):
        # Dataset paths
        self.dataset_paths = {
            'train': "augmented_data/train_data_with_paraphrases.json",
            'val_seen': "augmented_data/val_seen_data_with_paraphrases.json", 
            'val_unseen': "augmented_data/val_unseen_data_with_paraphrases.json"
        }
        
        # Backup paths
        self.backup_paths = {
            'train': "augmented_data/train_data_with_paraphrases_backup.json",
            'val_seen': "augmented_data/val_seen_data_with_paraphrases_backup.json", 
            'val_unseen': "augmented_data/val_unseen_data_with_paraphrases_backup.json"
        }
        
        # Pipeline components
        self.generation_pipeline = None
        self.validation_pipeline = None
        
        # Statistics
        self.stats = {
            'train': {'structure_issues_found': 0, 'structure_issues_fixed': 0, 'regeneration_failures': 0},
            'val_seen': {'structure_issues_found': 0, 'structure_issues_fixed': 0, 'regeneration_failures': 0},
            'val_unseen': {'structure_issues_found': 0, 'structure_issues_fixed': 0, 'regeneration_failures': 0}
        }
        
        logger.info("🔧 Structure Issue Fixer initialized")
        logger.info("🎯 Target: Fix dialog turns with incorrect paraphrase structure")
    
    def initialize_pipelines(self) -> bool:
        """Initialize generation and validation pipelines."""
        try:
            logger.info("🚀 Initializing pipelines...")
            
            # Initialize generation pipeline (GPUs 0-8)
            logger.info("📝 Loading ParaphraseGenerationPipeline...")
            self.generation_pipeline = ParaphraseGenerationPipeline()
            
            if not self.generation_pipeline.load_model():
                logger.error("❌ Failed to load generation model")
                return False
            
            # Initialize validation pipeline (GPU 9)
            logger.info("✅ Loading ValidationPipeline...")
            self.validation_pipeline = ValidationPipeline()
            
            # Move validation model to GPU 9 if available
            if torch.cuda.is_available() and torch.cuda.device_count() > 9:
                logger.info("📍 Moving validation model to GPU 9...")
                self.validation_pipeline.model = self.validation_pipeline.model.to('cuda:9')
                self.validation_pipeline.device = 'cuda:9'
            
            logger.info("✅ Pipelines initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Pipeline initialization failed: {e}")
            return False
    
    def fix_all_splits(self) -> Dict:
        """Fix structure issues in all dataset splits."""
        logger.info("🚀 Starting structure issue fixing for all splits...")
        
        results = {}
        
        for split in ['train', 'val_seen', 'val_unseen']:
            logger.info(f"\n{'='*60}")
            logger.info(f"🔧 Fixing {split.upper()} split")
            logger.info(f"{'='*60}")
            
            results[split] = self.fix_split(split)
        
        # Log final summary
        self._log_final_summary()
        
        return results
    
    def fix_split(self, split: str) -> Dict:
        """Fix structure issues in a specific split."""
        try:
            # Load dataset
            dataset_path = self.dataset_paths[split]
            
            if not Path(dataset_path).exists():
                logger.error(f"❌ Dataset not found: {dataset_path}")
                return {'success': False, 'error': f'File not found: {dataset_path}'}
            
            logger.info(f"📂 Loading {split} dataset...")
            with open(dataset_path, 'r') as f:
                episodes = json.load(f)
            
            # Create backup
            self._create_backup(episodes, split)
            
            # Identify and fix structure issues
            structure_issues = self._identify_structure_issues(episodes, split)
            
            if not structure_issues:
                logger.info(f"✅ No structure issues found in {split} split")
                return {'success': True, 'issues_fixed': 0}
            
            logger.info(f"🎯 Found {len(structure_issues)} structure issues to fix")
            
            # Fix each structure issue
            fixed_episodes = self._fix_structure_issues(episodes, structure_issues, split)
            
            # Save fixed dataset
            logger.info(f"💾 Saving fixed {split} dataset...")
            with open(dataset_path, 'w') as f:
                json.dump(fixed_episodes, f, indent=2)
            
            logger.info(f"✅ {split} split fixed successfully")
            return {
                'success': True, 
                'issues_found': self.stats[split]['structure_issues_found'],
                'issues_fixed': self.stats[split]['structure_issues_fixed'],
                'regeneration_failures': self.stats[split]['regeneration_failures']
            }
            
        except Exception as e:
            logger.error(f"❌ Error fixing {split} split: {e}")
            return {'success': False, 'error': str(e)}
    
    def _create_backup(self, episodes: List[Dict], split: str):
        """Create backup of original dataset."""
        backup_path = self.backup_paths[split]
        logger.info(f"💾 Creating backup: {backup_path}")
        
        with open(backup_path, 'w') as f:
            json.dump(episodes, f, indent=2)
        
        logger.info(f"✅ Backup created successfully")
    
    def _identify_structure_issues(self, episodes: List[Dict], split: str) -> List[Tuple[str, int, Dict]]:
        """Identify dialog turns with structure issues."""
        structure_issues = []
        
        for episode in episodes:
            episode_id = episode.get('episode_id', 'unknown')
            dialogs = episode.get('dialogs', [])
            
            for turn_idx, dialog in enumerate(dialogs):
                # Check if dialog has an answer and paraphrases
                answer = dialog.get('answer')
                paraphrases = dialog.get('paraphrases')
                
                if answer and answer.strip() and paraphrases:
                    positives = paraphrases.get('positives', [])
                    negatives = paraphrases.get('negatives', [])
                    
                    # Check if structure is incorrect (not 2P + 1N)
                    if len(positives) != 2 or len(negatives) != 1:
                        structure_issues.append((episode_id, turn_idx, dialog))
                        self.stats[split]['structure_issues_found'] += 1
                        
                        logger.info(f"📍 Structure issue: Episode {episode_id}, Turn {turn_idx} - "
                                  f"{len(positives)}P + {len(negatives)}N (expected 2P + 1N)")
        
        return structure_issues
    
    def _fix_structure_issues(self, episodes: List[Dict], structure_issues: List[Tuple], split: str) -> List[Dict]:
        """Fix identified structure issues by regenerating paraphrases."""
        # Create a mapping for quick lookup
        issue_map = {}
        for episode_id, turn_idx, dialog in structure_issues:
            if episode_id not in issue_map:
                issue_map[episode_id] = {}
            issue_map[episode_id][turn_idx] = dialog
        
        # Process episodes and fix issues
        fixed_episodes = []
        
        for episode in episodes:
            episode_id = episode.get('episode_id', 'unknown')
            
            if episode_id in issue_map:
                # This episode has structure issues - fix them
                fixed_episode = self._fix_episode_structure_issues(episode, issue_map[episode_id], split)
                fixed_episodes.append(fixed_episode)
            else:
                # No issues in this episode - keep as is
                fixed_episodes.append(episode)
        
        return fixed_episodes
    
    def _fix_episode_structure_issues(self, episode: Dict, turn_issues: Dict[int, Dict], split: str) -> Dict:
        """Fix structure issues in a specific episode."""
        episode_id = episode.get('episode_id', 'unknown')
        fixed_episode = episode.copy()
        fixed_dialogs = []
        
        for turn_idx, dialog in enumerate(episode.get('dialogs', [])):
            if turn_idx in turn_issues:
                # This turn has structure issues - regenerate paraphrases
                logger.info(f"🔄 Regenerating paraphrases for Episode {episode_id}, Turn {turn_idx}")
                
                answer = dialog.get('answer', '')
                fixed_dialog = self._regenerate_paraphrases_for_dialog(dialog, answer, episode_id, turn_idx, split)
                fixed_dialogs.append(fixed_dialog)
            else:
                # No issues - keep dialog as is
                fixed_dialogs.append(dialog)
        
        fixed_episode['dialogs'] = fixed_dialogs
        return fixed_episode
    
    def _regenerate_paraphrases_for_dialog(self, dialog: Dict, answer: str, episode_id: str, turn_idx: int, split: str) -> Dict:
        """Regenerate paraphrases for a specific dialog turn."""
        try:
            # Generate new paraphrases
            generation_result = self.generation_pipeline.generate_paraphrases(
                answer, 
                strategy="combined"
            )
            
            positives = generation_result.get('positives', [])
            negatives = generation_result.get('negatives', [])
            
            # Validate new paraphrases
            validation_result = self._validate_paraphrases(answer, positives, negatives)
            
            if validation_result['success']:
                # Update dialog with new paraphrases
                fixed_dialog = dialog.copy()
                fixed_dialog['paraphrases'] = {
                    'positives': positives,
                    'negatives': negatives,
                    'valid_positives': validation_result['valid_positives'],
                    'valid_negatives': validation_result['valid_negatives'],
                    'validation_analysis': validation_result['validation_report']
                }
                
                self.stats[split]['structure_issues_fixed'] += 1
                
                # Verify the fix
                new_pos_count = len(positives)
                new_neg_count = len(negatives)
                
                if new_pos_count == 2 and new_neg_count == 1:
                    logger.info(f"✅ Episode {episode_id}, Turn {turn_idx}: Fixed structure ({new_pos_count}P + {new_neg_count}N)")
                else:
                    logger.warning(f"⚠️ Episode {episode_id}, Turn {turn_idx}: Regeneration still has structure issue ({new_pos_count}P + {new_neg_count}N)")
                
                return fixed_dialog
            else:
                logger.error(f"❌ Episode {episode_id}, Turn {turn_idx}: Regeneration failed - keeping original")
                self.stats[split]['regeneration_failures'] += 1
                return dialog
                
        except Exception as e:
            logger.error(f"❌ Episode {episode_id}, Turn {turn_idx}: Regeneration error - {e}")
            self.stats[split]['regeneration_failures'] += 1
            return dialog
    
    def _validate_paraphrases(self, answer: str, positives: List[str], negatives: List[str]) -> Dict:
        """Validate paraphrases using GPU 9."""
        try:
            # Ensure we're using GPU 9
            if torch.cuda.is_available() and torch.cuda.device_count() > 9:
                torch.cuda.set_device(9)
                torch.cuda.empty_cache()
            
            validation_report = {
                'original_answer': answer,
                'valid_positives': [],
                'valid_negatives': [],
                'validation_details': {
                    'positive_results': [],
                    'negative_results': []
                }
            }
            
            # Validate positives
            for positive in positives:
                result = self.validation_pipeline.validate_positive_paraphrase(answer, positive)
                validation_report['validation_details']['positive_results'].append(result)
                
                if result['is_valid']:
                    validation_report['valid_positives'].append(positive)
            
            # Validate negatives
            for negative in negatives:
                result = self.validation_pipeline.validate_negative_paraphrase(answer, negative)
                validation_report['validation_details']['negative_results'].append(result)
                
                if result['is_valid']:
                    validation_report['valid_negatives'].append(negative)
            
            # Cleanup GPU 9
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {
                'success': True,
                'valid_positives': validation_report['valid_positives'],
                'valid_negatives': validation_report['valid_negatives'],
                'validation_report': validation_report
            }
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {'success': False, 'error': str(e)}
    
    def _log_final_summary(self):
        """Log final summary of structure fixes."""
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 STRUCTURE FIX SUMMARY")
        logger.info(f"{'='*60}")
        
        total_found = sum(self.stats[split]['structure_issues_found'] for split in self.stats)
        total_fixed = sum(self.stats[split]['structure_issues_fixed'] for split in self.stats)
        total_failures = sum(self.stats[split]['regeneration_failures'] for split in self.stats)
        
        for split in ['train', 'val_seen', 'val_unseen']:
            split_stats = self.stats[split]
            if split_stats['structure_issues_found'] > 0:
                logger.info(f"\n🎯 {split.upper()} SPLIT:")
                logger.info(f"  Structure issues found: {split_stats['structure_issues_found']}")
                logger.info(f"  Structure issues fixed: {split_stats['structure_issues_fixed']}")
                logger.info(f"  Regeneration failures: {split_stats['regeneration_failures']}")
                
                if split_stats['structure_issues_found'] > 0:
                    fix_rate = split_stats['structure_issues_fixed'] / split_stats['structure_issues_found']
                    logger.info(f"  Fix success rate: {fix_rate:.2%}")
        
        logger.info(f"\n🎯 OVERALL TOTALS:")
        logger.info(f"  Total structure issues found: {total_found}")
        logger.info(f"  Total structure issues fixed: {total_fixed}")
        logger.info(f"  Total regeneration failures: {total_failures}")
        
        if total_found > 0:
            overall_fix_rate = total_fixed / total_found
            logger.info(f"  Overall fix success rate: {overall_fix_rate:.2%}")
        
        if total_fixed == total_found:
            logger.info(f"🎉 ALL STRUCTURE ISSUES FIXED SUCCESSFULLY!")
        elif total_fixed > 0:
            logger.info(f"✅ Most structure issues fixed ({total_fixed}/{total_found})")
        else:
            logger.warning(f"⚠️ No structure issues were fixed")

def main():
    """Run the structure issue fixer."""
    fixer = StructureIssueFixer()
    
    try:
        # Initialize pipelines
        if not fixer.initialize_pipelines():
            logger.error("❌ Pipeline initialization failed")
            return
        
        # Fix all splits
        results = fixer.fix_all_splits()
        
        # Check overall success
        all_successful = all(result.get('success', False) for result in results.values())
        
        if all_successful:
            total_fixed = sum(result.get('issues_fixed', 0) for result in results.values())
            logger.info(f"✅ Structure fixing completed successfully - {total_fixed} issues fixed")
        else:
            failed_splits = [split for split, result in results.items() if not result.get('success', False)]
            logger.error(f"❌ Structure fixing failed for splits: {failed_splits}")
        
    except Exception as e:
        logger.error(f"❌ Structure fixing failed: {e}")
    finally:
        # Cleanup GPUs
        if hasattr(fixer, 'generation_pipeline') and fixer.generation_pipeline:
            import gc
            gc.collect()
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    try:
                        with torch.cuda.device(i):
                            torch.cuda.empty_cache()
                    except Exception:
                        pass

if __name__ == "__main__":
    main() 