# Testing Commands for Headless Server

## Quick Test (Start Here)

```bash
# Navigate to data directory
cd AnsweringAgent/src/data

# Quick functionality test
python run_simple_test.py
```

**Expected Output:**
```
✅ Import successful
🔄 Initializing Mixtral...
✅ Initialization successful
🔄 Testing: Fly to the white building on your right.
✅ Success! Paraphrase: [Generated paraphrase]
🎉 Basic functionality confirmed!
```

---

## Comprehensive Test (If Quick Test Passes)

```bash
# Full test suite
python test_simple_mixtral.py
```

**Expected Output:**
```
=== Environment Testing ===
Python version: 3.x.x
PyTorch version: x.x.x
CUDA available: True
CUDA device count: 10
Device 0: NVIDIA GeForce RTX 2080 Ti
  Memory: 11.00 GB
[... 9 more devices ...]

=== Import Testing ===
✅ Successfully imported SimpleMixtralParaphraser

=== Initialization Testing ===
✅ Successfully initialized SimpleMixtralParaphraser

=== GPU Memory Testing ===
GPU 0 memory:
  Allocated: X.XX GB
  Cached: Y.YY GB
[... memory info ...]

=== Paraphrasing Testing ===
Test 1: Fly to the white building on your right.
✅ Paraphrase: [Generated paraphrase]
[... 4 more tests ...]

=== Test Summary ===
Total paraphrasing tests: 5
Successful tests: 5
Failed tests: 0
🎉 All tests passed!
```

---

## Direct Script Test (Alternative)

```bash
# Test the original script directly
python simple_mixtral_paraphrasing.py
```

---

## Installation Check (If Import Fails)

```bash
# Check if required packages are installed
pip install torch transformers accelerate bitsandbytes

# Or install from requirements
pip install -r requirements_simple_mixtral.txt
```

---

## Common Issues & Solutions

### Issue 1: CUDA Out of Memory
**Symptom:** `RuntimeError: CUDA out of memory`
**Solution:** Model distributed across multiple GPUs automatically, but if still occurs:
- Check GPU memory usage: `nvidia-smi`
- Reduce batch size or use CPU fallback

### Issue 2: Model Download Issues
**Symptom:** `OSError: Unable to load weights`
**Solution:** Check internet connection and disk space
- Model is ~90GB, ensure sufficient space
- Check Hugging Face cache: `~/.cache/huggingface/`

### Issue 3: Import Errors
**Symptom:** `ModuleNotFoundError`
**Solution:** Install missing dependencies
```bash
pip install torch transformers accelerate bitsandbytes
```

---

## Testing Checklist

- [ ] Quick test passes (`run_simple_test.py`)
- [ ] All 10 GPUs detected
- [ ] Model loads without memory errors
- [ ] At least 1 successful paraphrase generated
- [ ] No import errors
- [ ] Reasonable paraphrase quality (UAV navigation context preserved)

---

## Next Steps After Successful Testing

1. **Share Results:** Copy test output and share with me
2. **Identify Issues:** Note any failures or quality concerns
3. **Enhancement Planning:** Choose first enhancement from roadmap
4. **Integration:** Move to pipeline integration or UAV specialization

**Ready for your validation!** 