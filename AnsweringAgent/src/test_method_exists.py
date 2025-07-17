#!/usr/bin/env python3
"""Test script to check if preprocess_and_save method exists."""

from data.dataset import AnsweringDataset

# Check if the method exists
if hasattr(AnsweringDataset, 'preprocess_and_save'):
    print("✅ preprocess_and_save method exists")
    print(f"Method: {AnsweringDataset.preprocess_and_save}")
else:
    print("❌ preprocess_and_save method does not exist")
    print("Available methods:")
    for attr in dir(AnsweringDataset):
        if not attr.startswith('_'):
            print(f"  - {attr}") 