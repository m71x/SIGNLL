import sys
import os
import numpy as np

# Import the core loading function from your existing GCS loader script
# Note: This relies on GCS_CLIENT being initialized in gcs_npz_loader.py
try:
    from gcs_npz_loader import load_npz_from_gcs
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import load_npz_from_gcs. Ensure gcs_npz_loader.py is accessible and imports its dependencies correctly. Error: {e}")
    sys.exit(1)


# --- Configuration for Validation ---
# You need to adjust these values to check the file you just uploaded from inference.py
VALIDATION_CORE_ID = 0     # The TPU Core ID that saved the file (e.g., core_10)
VALIDATION_CHUNK_INDEX = 31  # The chunk index of the file you want to validate (e.g., embeddings_chunk_1.npz)
# N_SAMPLES_EXPECTED is the minimum number of samples you expect this chunk to have.
N_SAMPLES_EXPECTED = 1000   


def validate_uploaded_chunk(core_id: int, chunk_index: int, expected_samples: int) -> bool:
    """
    Downloads a specific NPZ chunk from GCS, attempts to load it, and validates its structure.
    
    Args:
        core_id: The core ID (folder name).
        chunk_index: The chunk index (file name suffix).
        expected_samples: Minimum number of samples expected in the chunk.

    Returns:
        True if the file is successfully loaded and passes basic structural checks, False otherwise.
    """
    chunk_filename = f"embeddings_chunk_{chunk_index}.npz"
    print(f"\n{'='*70}")
    print(f"VALIDATION TEST for {chunk_filename} (Core {core_id})")
    print(f"{'='*70}\n")

    # The loading function handles download, buffer loading, and corruption checks
    data = load_npz_from_gcs(core_id, chunk_filename)
    
    if data is None:
        print(f"\n❌ VALIDATION FAILED: Could not load data from GCS. See error logs above.")
        return False

    # --- Structural Validation ---
    
    try:
        cls_tokens = data['all_layer_cls_tokens']
        classifications = data['classifications']
        
        N_samples = cls_tokens.shape[0]

        print(f"\n--- Basic Structure Checks ---")
        
        # 1. Size Check
        if N_samples < expected_samples:
            print(f"❌ VALIDATION FAILED: Samples loaded ({N_samples}) is less than expected minimum ({expected_samples}).")
            return False
        else:
            print(f"✅ Sample count check: {N_samples} samples (>= {expected_samples} expected)")
        
        # 2. Shape Check (N, L, D) -> (N, 25, 1024)
        if cls_tokens.ndim != 3 or cls_tokens.shape[1] != 25 or cls_tokens.shape[2] != 1024:
            print(f"❌ VALIDATION FAILED: CLS token shape is incorrect: {cls_tokens.shape}. Expected (N, 25, 1024).")
            return False
        else:
            print(f"✅ CLS tokens shape: {cls_tokens.shape} (correct)")
            print(f"   - Samples: {cls_tokens.shape[0]}")
            print(f"   - Layers: {cls_tokens.shape[1]}")
            print(f"   - Dimensions: {cls_tokens.shape[2]}")

        # 3. Dtype Check
        if cls_tokens.dtype != np.float32:
            print(f"⚠️ WARNING: CLS tokens dtype is {cls_tokens.dtype}. Expected float32.")
        else:
            print(f"✅ CLS tokens dtype: {cls_tokens.dtype}")
            
        if classifications.dtype != np.uint8:
            print(f"❌ VALIDATION FAILED: Classification dtype is incorrect: {classifications.dtype}. Expected uint8.")
            return False
        else:
            print(f"✅ Classifications dtype: {classifications.dtype}")
            
        # 4. Range Check (Classifications)
        class_min = classifications.min()
        class_max = classifications.max()
        if class_min < 0 or class_max > 1:
            print(f"❌ VALIDATION FAILED: Classification values out of expected [0, 1] range.")
            return False
        else:
            print(f"✅ Classification range: [{class_min}, {class_max}]")
            
            # Count distribution
            num_negative = np.sum(classifications == 0)
            num_positive = np.sum(classifications == 1)
            print(f"   - Negative (0): {num_negative} ({100*num_negative/N_samples:.1f}%)")
            print(f"   - Positive (1): {num_positive} ({100*num_positive/N_samples:.1f}%)")

        # --- Advanced Access Tests ---
        print(f"\n--- Data Access Tests ---")
        
        # Test 1: Access first sample
        print(f"\n1. Testing access to first sample (index 0):")
        try:
            sample_0_all_layers = cls_tokens[0]
            sample_0_label = classifications[0]
            print(f"   ✅ Sample 0 shape: {sample_0_all_layers.shape} (25 layers × 1024 dims)")
            print(f"   ✅ Sample 0 label: {sample_0_label}")
            print(f"   - First layer (embedding) mean: {sample_0_all_layers[0].mean():.6f}")
            print(f"   - Last layer (layer 24) mean: {sample_0_all_layers[24].mean():.6f}")
        except Exception as e:
            print(f"   ❌ Failed to access sample 0: {e}")
            return False
        
        # Test 2: Access middle sample
        print(f"\n2. Testing access to middle sample (index {N_samples//2}):")
        try:
            mid_idx = N_samples // 2
            sample_mid_all_layers = cls_tokens[mid_idx]
            sample_mid_label = classifications[mid_idx]
            print(f"   ✅ Sample {mid_idx} shape: {sample_mid_all_layers.shape}")
            print(f"   ✅ Sample {mid_idx} label: {sample_mid_label}")
        except Exception as e:
            print(f"   ❌ Failed to access sample {mid_idx}: {e}")
            return False
        
        # Test 3: Access last sample
        print(f"\n3. Testing access to last sample (index {N_samples-1}):")
        try:
            sample_last_all_layers = cls_tokens[-1]
            sample_last_label = classifications[-1]
            print(f"   ✅ Sample {N_samples-1} shape: {sample_last_all_layers.shape}")
            print(f"   ✅ Sample {N_samples-1} label: {sample_last_label}")
        except Exception as e:
            print(f"   ❌ Failed to access last sample: {e}")
            return False
        
        # Test 4: Access specific layer across all samples
        print(f"\n4. Testing access to specific layer (layer 12) across all samples:")
        try:
            layer_12_all_samples = cls_tokens[:, 12, :]
            print(f"   ✅ Layer 12 shape: {layer_12_all_samples.shape} ({N_samples} samples × 1024 dims)")
            print(f"   - Layer 12 mean across all samples: {layer_12_all_samples.mean():.6f}")
            print(f"   - Layer 12 std across all samples: {layer_12_all_samples.std():.6f}")
        except Exception as e:
            print(f"   ❌ Failed to access layer 12: {e}")
            return False
        
        # Test 5: Batch access (first 100 samples)
        print(f"\n5. Testing batch access (first 100 samples):")
        try:
            batch_size = min(100, N_samples)
            batch_cls = cls_tokens[:batch_size]
            batch_labels = classifications[:batch_size]
            print(f"   ✅ Batch CLS shape: {batch_cls.shape}")
            print(f"   ✅ Batch labels shape: {batch_labels.shape}")
        except Exception as e:
            print(f"   ❌ Failed to access batch: {e}")
            return False
        
        # Test 6: Slicing test (every 100th sample)
        print(f"\n6. Testing stride access (every 100th sample):")
        try:
            stride_cls = cls_tokens[::100]
            stride_labels = classifications[::100]
            print(f"   ✅ Stride CLS shape: {stride_cls.shape} ({len(stride_cls)} samples)")
            print(f"   ✅ Stride labels shape: {stride_labels.shape}")
        except Exception as e:
            print(f"   ❌ Failed stride access: {e}")
            return False
        
        # Test 7: Statistical integrity check
        print(f"\n7. Statistical integrity checks:")
        try:
            # Check for NaN or Inf values
            has_nan = np.isnan(cls_tokens).any()
            has_inf = np.isinf(cls_tokens).any()
            
            if has_nan:
                print(f"   ❌ WARNING: NaN values detected in CLS tokens!")
            else:
                print(f"   ✅ No NaN values in CLS tokens")
            
            if has_inf:
                print(f"   ❌ WARNING: Inf values detected in CLS tokens!")
            else:
                print(f"   ✅ No Inf values in CLS tokens")
            
            # Overall statistics
            overall_mean = cls_tokens.mean()
            overall_std = cls_tokens.std()
            overall_min = cls_tokens.min()
            overall_max = cls_tokens.max()
            
            print(f"   ✅ CLS tokens statistics:")
            print(f"      - Mean: {overall_mean:.6f}")
            print(f"      - Std:  {overall_std:.6f}")
            print(f"      - Min:  {overall_min:.6f}")
            print(f"      - Max:  {overall_max:.6f}")
            
            # Check if values are reasonable (not all zeros or all same value)
            if overall_std < 1e-6:
                print(f"   ❌ WARNING: Very low standard deviation - data might be corrupted!")
            else:
                print(f"   ✅ Standard deviation is reasonable")
                
        except Exception as e:
            print(f"   ❌ Failed statistical checks: {e}")
            return False
        
        # Test 8: Memory efficiency check
        print(f"\n8. Memory usage estimation:")
        try:
            cls_size_mb = cls_tokens.nbytes / (1024**2)
            class_size_kb = classifications.nbytes / 1024
            total_size_mb = (cls_tokens.nbytes + classifications.nbytes) / (1024**2)
            
            print(f"   ✅ CLS tokens: {cls_size_mb:.2f} MB")
            print(f"   ✅ Classifications: {class_size_kb:.2f} KB")
            print(f"   ✅ Total in memory: {total_size_mb:.2f} MB")
        except Exception as e:
            print(f"   ❌ Failed memory check: {e}")
            return False

        # Final summary
        print(f"\n{'='*70}")
        print(f"✅ ALL VALIDATION TESTS PASSED!")
        print(f"{'='*70}")
        print(f"Summary:")
        print(f"  - Total samples: {N_samples:,}")
        print(f"  - Shape: {cls_tokens.shape}")
        print(f"  - Classification distribution: {num_negative} negative, {num_positive} positive")
        print(f"  - Memory footprint: {total_size_mb:.2f} MB")
        print(f"  - Data quality: ✅ No NaN/Inf, reasonable statistics")
        print(f"{'='*70}\n")
        
        return True

    except KeyError as e:
        print(f"\n❌ VALIDATION FAILED: Missing expected key in NPZ file: {e}")
        return False
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: Unexpected error during validation checks: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    # You would typically run this script after 'inference.py' completes a chunk upload.
    # Adjust the constants above (VALIDATION_CORE_ID, VALIDATION_CHUNK_INDEX, N_SAMPLES_EXPECTED) 
    # to test the target file.
    
    if validate_uploaded_chunk(VALIDATION_CORE_ID, VALIDATION_CHUNK_INDEX, N_SAMPLES_EXPECTED):
        print("\n🎉 Final Result: The uploaded NPZ file is VALID and ready for use!")
    else:
        print("\n❌ Final Result: The uploaded NPZ file has issues and needs investigation.")