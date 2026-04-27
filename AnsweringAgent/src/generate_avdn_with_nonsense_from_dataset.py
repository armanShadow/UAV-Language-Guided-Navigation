import os
import json
import argparse
import random
from typing import Dict, List, Tuple

# Suppress warnings for cleaner output
import warnings

warnings.filterwarnings("ignore")


class AVDNDatasetNonsenseReplacer:
    """Replace AVDN instructions with random 'nonsense' drawn from the same dataset."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.pool: List[str] = []
        self.pool_usage: Dict[str, int] = {}

    @staticmethod
    def _repo_root() -> str:
        # This file lives at AnsweringAgent/src/..., so repo root is two levels up.
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    def load_avdn_data(self, split: str) -> List[Dict]:
        """Load original AVDN dataset for a specific split."""
        data_file = os.path.join(
            self._repo_root(),
            "Aerial-Vision-and-Dialog-Navigation",
            "datasets",
            "AVDN",
            "annotations",
            f"{split}_data.json",
        )
        print(f"Loading AVDN data from: {data_file}")

        with open(data_file, "r") as f:
            data = json.load(f)

        print(f"Loaded {len(data)} samples from {split} split")
        return data

    @staticmethod
    def _extract_answer_text(instruction: str) -> str:
        """Extract the [INS] answer text from an AVDN instruction string."""
        ins_start = instruction.find("[INS]")
        if ins_start != -1:
            return instruction[ins_start + 5 :].strip()
        return instruction.strip()

    def build_pool_from_split(self, avdn_data: List[Dict], split: str) -> None:
        """Build the replacement pool from a split's original data."""
        answers: List[str] = []
        for sample in avdn_data:
            instr = sample.get("instructions", "")
            if "[QUE]" in instr and "[INS]" in instr:
                ans = self._extract_answer_text(instr)
                if ans:
                    answers.append(ans)

        if not answers:
            raise RuntimeError(
                f"No Q&A answers found to build a pool for split={split}. "
                "Expected instructions containing both [QUE] and [INS]."
            )

        self.pool = answers
        self.pool_usage = {a: 0 for a in self.pool}
        print(f"Built replacement pool of {len(self.pool)} answers from {split} split")

    def find_similar_length_replacement(self, original_answer: str, split: str) -> str:
        """Find a replacement answer with similar length, respecting reuse constraints."""
        if not self.pool:
            raise RuntimeError("Replacement pool is empty. Call build_pool_from_split first.")

        original_length = len(original_answer)
        max_reuse = 5 if split == "train" else 2

        min_length = int(original_length * 0.8)
        max_length = int(original_length * 1.2)

        candidates: List[str] = []
        for ans in self.pool:
            if ans == original_answer:
                continue
            if min_length <= len(ans) <= max_length and self.pool_usage.get(ans, 0) < max_reuse:
                candidates.append(ans)

        if candidates:
            selected = random.choice(candidates)
            self.pool_usage[selected] = self.pool_usage.get(selected, 0) + 1
            return selected

        available = [
            ans
            for ans in self.pool
            if ans != original_answer and self.pool_usage.get(ans, 0) < max_reuse
        ]
        if available:
            selected = random.choice(available)
            self.pool_usage[selected] = self.pool_usage.get(selected, 0) + 1
            return selected

        # If everything hit max usage (or only the original remains), fall back to any non-identical.
        non_identical = [ans for ans in self.pool if ans != original_answer]
        selected = random.choice(non_identical) if non_identical else random.choice(self.pool)
        self.pool_usage[selected] = self.pool_usage.get(selected, 0) + 1
        return selected

    def update_avdn_instruction(self, avdn_sample: Dict, replacement_answer: str) -> Dict:
        """Update AVDN sample with replacement answer while preserving dialog structure."""
        new_sample = avdn_sample.copy()

        original_instruction = avdn_sample["instructions"]
        if "[QUE]" in original_instruction and "[INS]" in original_instruction:
            parts = original_instruction.split("[INS]")
            question_part = parts[0].replace("[QUE]", "").strip()
            new_instruction = f"[QUE] {question_part} [INS] {replacement_answer}"
        else:
            if "[INS]" in original_instruction:
                new_instruction = f"[INS] {replacement_answer}"
            else:
                new_instruction = replacement_answer

        new_sample["instructions"] = new_instruction
        new_sample["_debug_info"] = {
            "original_instruction": original_instruction,
            "replacement_answer": replacement_answer,
            "original_length": len(original_instruction),
            "replacement_length": len(replacement_answer),
            "length_ratio": (
                len(replacement_answer) / len(original_instruction) if len(original_instruction) > 0 else 0
            ),
        }
        return new_sample

    def process_avdn_sample(self, avdn_sample: Dict, avdn_index: int, split: str) -> Dict:
        """Process a single AVDN sample and replace its answer with a random dataset answer."""
        map_name = avdn_sample["map_name"]
        route_index = avdn_sample["route_index"]

        original_instruction = avdn_sample["instructions"]

        if "[QUE]" not in original_instruction or "[INS]" not in original_instruction:
            if avdn_index < 10:
                print(f"⚠️  Skipping non-Q&A sample {avdn_index} ({map_name}_{route_index})")
            return avdn_sample

        original_answer = self._extract_answer_text(original_instruction)
        replacement = self.find_similar_length_replacement(original_answer, split)
        new_sample = self.update_avdn_instruction(avdn_sample, replacement)

        if avdn_index < 3:
            print(f"\n🔍 Dataset-derived Replacement Debug for sample {avdn_index}:")
            print(f"   Map: {map_name}, Route: {route_index}")
            print(f"   Original: {original_answer}")
            print(f"   Replacement: {replacement}")
            print(f"   Length ratio: {len(replacement)/len(original_answer):.2f}" if len(original_answer) > 0 else "   Length ratio: n/a")
            print(f"   Usage count: {self.pool_usage.get(replacement, 0)}")

        return new_sample

    def update_pre_dialogs(self, data: List[Dict]) -> List[Dict]:
        """Update pre_dialogs for all samples based on replaced instructions."""
        print("🔄 Starting pre_dialogs update with dataset-derived replacements...")

        episodes: Dict[str, List[Tuple[int, Dict]]] = {}
        for i, sample in enumerate(data):
            map_name = sample["map_name"]
            route_index = sample["route_index"]

            episode_key = route_index.rsplit("_", 1)[0]
            full_episode_key = f"{map_name}_{episode_key}"

            if full_episode_key not in episodes:
                episodes[full_episode_key] = []
            episodes[full_episode_key].append((i, sample))

        for episode_key in episodes:
            episodes[episode_key].sort(key=lambda x: int(x[1]["route_index"].split("_")[-1]))

        print(f"📊 Found {len(episodes)} episodes to process")

        updated_data = [sample.copy() for sample in data]

        for episode_key, episode_samples in episodes.items():
            if len(episode_samples) > 1:
                print(f"🔄 Processing episode {episode_key} with {len(episode_samples)} turns")
                for turn_idx, (sample_idx, sample) in enumerate(episode_samples):
                    if turn_idx > 0:
                        new_instruction = sample["instructions"]

                        if "_debug_info" in updated_data[sample_idx]:
                            updated_data[sample_idx].pop("_debug_info")

                        for future_turn_idx in range(turn_idx + 1, len(episode_samples)):
                            future_sample_idx = episode_samples[future_turn_idx][0]
                            if turn_idx < len(updated_data[future_sample_idx]["pre_dialogs"]):
                                updated_data[future_sample_idx]["pre_dialogs"][turn_idx] = new_instruction

        return updated_data

    def process_split(self, split: str, sample_ratio: float = 1.0) -> Tuple[List[Dict], Dict]:
        """Process an entire split with dataset-derived replacements."""
        print(f"\n🚀 Processing {split} split with dataset-derived replacements...")

        avdn_data_full = self.load_avdn_data(split)
        self.build_pool_from_split(avdn_data_full, split)

        avdn_data = avdn_data_full
        if sample_ratio < 1.0:
            num_samples = int(len(avdn_data_full) * sample_ratio)
            avdn_data = avdn_data_full[:num_samples]
            print(f"📊 Sampled {num_samples}/{len(avdn_data_full)} samples ({sample_ratio*100:.1f}%)")

        processed_data: List[Dict] = []
        successful_replacements = 0
        total_qa_samples = 0

        for i, sample in enumerate(avdn_data):
            try:
                processed_sample = self.process_avdn_sample(sample, i, split)
                processed_data.append(processed_sample)

                if processed_sample["instructions"] != sample["instructions"]:
                    successful_replacements += 1

                if "[QUE]" in sample["instructions"] and "[INS]" in sample["instructions"]:
                    total_qa_samples += 1
            except Exception as e:
                print(f"⚠️ Error processing sample {i}: {e}")
                processed_data.append(sample)

        processed_data = self.update_pre_dialogs(processed_data)

        used_answers = [a for a, c in self.pool_usage.items() if c > 0]
        total_usage = sum(self.pool_usage.values())
        avg_usage = total_usage / len(used_answers) if used_answers else 0
        max_usage = max(self.pool_usage.values()) if self.pool_usage else 0
        min_usage = min(self.pool_usage.values()) if self.pool_usage else 0

        print(f"\n📊 {split.upper()} DATASET-DERIVED REPLACEMENT STATISTICS:")
        print(f"Total Samples: {len(processed_data)}")
        print(f"Q&A Samples: {total_qa_samples}")
        print(f"Successful Replacements: {successful_replacements}")
        print(
            f"Success Rate: {successful_replacements/total_qa_samples*100:.1f}%"
            if total_qa_samples > 0
            else "No Q&A samples found"
        )
        print(f"\n📈 POOL USAGE STATISTICS:")
        print(f"Total Answers Used: {len(used_answers)}/{len(self.pool)}")
        print(f"Average Usage per Answer: {avg_usage:.2f}")
        print(f"Maximum Usage: {max_usage}")
        print(f"Minimum Usage: {min_usage}")
        print(f"Total Usage Count: {total_usage}")

        examples_shown = 0
        for original, processed in zip(avdn_data, processed_data):
            if examples_shown >= 3:
                break
            if processed["instructions"] != original["instructions"]:
                examples_shown += 1
                print(f"\nGenerated Example {examples_shown}:")
                print(f"Map: {original['map_name']}, Route: {original['route_index']}")
                print(f"Original: {original['instructions']}")
                print(f"Replacement: {processed['instructions']}")
                if "_debug_info" in processed:
                    print(f"Length ratio: {processed['_debug_info']['length_ratio']:.2f}")
                print("-" * 80)

        return processed_data, {
            "total_samples": len(processed_data),
            "qa_samples": total_qa_samples,
            "successful_replacements": successful_replacements,
            "success_rate": successful_replacements / total_qa_samples if total_qa_samples > 0 else 0.0,
            "usage_stats": {
                "used_answers": len(used_answers),
                "total_answers": len(self.pool),
                "avg_usage": avg_usage,
                "max_usage": max_usage,
                "min_usage": min_usage,
                "total_usage": total_usage,
            },
        }

    def save_processed_data(self, data: List[Dict], split: str) -> None:
        output_file = os.path.join(self.output_dir, f"{split}_data.json")
        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(data)} samples to {output_file}")

    def process_all_splits(self, splits: List[str], sample_ratio: float = 1.0) -> None:
        overall_metrics: Dict[str, Dict] = {}

        for split in splits:
            processed_data, split_metrics = self.process_split(split, sample_ratio)
            self.save_processed_data(processed_data, split)
            overall_metrics[split] = split_metrics

        print(f"\n🎯 OVERALL DATASET-DERIVED REPLACEMENT SUMMARY:")
        print("=" * 80)
        for split, metrics in overall_metrics.items():
            print(f"{split.upper()}:")
            print(
                f"  📊 Replacement: {metrics['successful_replacements']}/{metrics['qa_samples']} "
                f"Q&A samples replaced ({metrics['success_rate']*100:.1f}% success)"
            )
            print(
                f"  📈 Usage: {metrics['usage_stats']['used_answers']}/"
                f"{metrics['usage_stats']['total_answers']} answers used"
            )
            print(f"  📊 Average usage: {metrics['usage_stats']['avg_usage']:.2f} per answer")
            print(
                f"  📊 Max usage: {metrics['usage_stats']['max_usage']}, "
                f"Min usage: {metrics['usage_stats']['min_usage']}"
            )
            print()

        print(f"\n✅ Dataset-derived replacement completed for all splits!")
        print(f"📁 Generated datasets saved to: {self.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replace AVDN [INS] answers with random answers sampled from the same AVDN split"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./generated_avdn_dataset_nonsense_from_dataset",
        help="Output directory for generated dataset",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val_seen", "val_unseen"],
        help="Dataset splits to process",
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=1.0,
        help="Ratio of dataset to sample (default: 1.0 = 100%)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    args = parser.parse_args()

    print("🚀 AVDN Dataset Generation with Dataset-Derived 'Nonsense' Instructions")
    print(f"Output Dir: {args.output_dir}")
    print(f"Splits: {args.splits}")
    print(f"Sample Ratio: {args.sample_ratio}")

    random.seed(args.seed)

    replacer = AVDNDatasetNonsenseReplacer(output_dir=args.output_dir)

    print("\n🚀 Starting AVDN dataset generation with dataset-derived replacements...")
    replacer.process_all_splits(splits=args.splits, sample_ratio=args.sample_ratio)

    print("\n✅ AVDN dataset generation with dataset-derived replacements complete!")
    print(f"📁 Generated dataset saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
