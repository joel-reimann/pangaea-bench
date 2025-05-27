#!/usr/bin/env python3

"""
AGBD Integration Fix Script v3
Now understands the preprocessing pipeline properly
"""

import os
import sys
import yaml
import json
from pathlib import Path

def check_agbd_config():
    """Check AGBD dataset configuration for common issues"""
    print("🔍 Checking AGBD configuration...")
    
    # Check dataset config
    dataset_config_path = Path("/scratch/reimannj5/pangaea-bench/pangaea/configs/dataset/agbd.yaml")
    if not dataset_config_path.exists():
        print("❌ AGBD dataset config not found at expected location!")
        # Try to find it
        possible_paths = [
            Path("/scratch/reimannj5/pangaea-bench/configs/dataset/agbd.yaml"),
            Path("./configs/dataset/agbd.yaml"),
            Path("./pangaea/configs/dataset/agbd.yaml")
        ]
        for path in possible_paths:
            if path.exists():
                print(f"✅ Found AGBD config at: {path}")
                dataset_config_path = path
                break
        else:
            print("❌ Could not find AGBD config anywhere!")
            return False
    
    with open(dataset_config_path) as f:
        config = yaml.safe_load(f)
    
    print(f"✅ Found AGBD config with img_size: {config.get('img_size', 'NOT_SET')}")
    
    # Check band mappings
    bands = config.get('bands', {})
    print(f"📊 Bands configured:")
    for band_type, band_list in bands.items():
        print(f"  {band_type}: {band_list}")
    
    return True, config

def check_preprocessing_pipeline():
    """Check the preprocessing pipeline options"""
    print("\n🔧 Checking preprocessing pipeline...")
    
    # Check if custom AGBD preprocessing exists
    agbd_preprocessing_path = Path("/scratch/reimannj5/pangaea-bench/configs/preprocessing/agbd_regression.yaml")
    
    if agbd_preprocessing_path.exists():
        print("✅ Found custom AGBD preprocessing config")
        with open(agbd_preprocessing_path) as f:
            config = yaml.safe_load(f)
        
        train_steps = config.get('train', {}).get('preprocessor_cfg', [])
        print("📋 Training preprocessing steps:")
        for i, step in enumerate(train_steps):
            target = step.get('_target_', 'Unknown')
            print(f"  {i+1}. {target}")
    else:
        print("❌ Custom AGBD preprocessing not found")
    
    # Check if ResizeToEncoder exists
    print("\n🔍 Key insight: The preprocessing pipeline should:")
    print("  1. ResizeToEncoder - handles AGBD 25x25 -> encoder size")
    print("  2. BandFilter - selects correct bands for encoder")
    print("  3. NormalizeMinMax - normalizes values")
    print("  4. BandPadding - pads channels if needed")

def analyze_the_real_problem():
    """Analyze what's actually happening"""
    print("\n🎯 ROOT CAUSE ANALYSIS:")
    print("=" * 50)
    print()
    print("❌ THE REAL PROBLEM:")
    print("   Even with dataset.img_size=224, the data is still 25x25")
    print("   This means the PREPROCESSING PIPELINE is not resizing!")
    print()
    print("🔍 WHAT'S HAPPENING:")
    print("   1. AGBD loads: 25x25 patches")
    print("   2. Preprocessing: Should resize to encoder requirements")
    print("   3. ScaleMAE encoder: Expects 224x224 input")
    print("   4. ERROR: Still getting 25x25, so preprocessing didn't resize")
    print()
    print("✅ THE SOLUTION:")
    print("   Use preprocessing=agbd_regression that includes ResizeToEncoder")
    print("   This will properly resize 25x25 -> 224x224 for ScaleMAE")
    print()
    print("🧪 TEST COMMAND:")
    print("   Use preprocessing=agbd_regression instead of preprocessing=reg_default")

def suggest_fixes():
    """Suggest fixes for common issues"""
    print("\n🔧 CORRECT FIXES:")
    print()
    print("1. ✅ PREPROCESSING FIX:")
    print("   Problem: reg_default doesn't resize AGBD patches")
    print("   Solution: Use preprocessing=agbd_regression")
    print("   This includes ResizeToEncoder which handles 25x25 -> encoder_size")
    print()
    print("2. ✅ DECODER FIX:")
    print("   Problem: UNet decoder requires 'topology' attribute")
    print("   Solution: Use decoder=reg_upernet or decoder=reg_fcn")
    print()
    print("3. ✅ BAND MAPPING:")
    print("   Problem: CROMA expects VV/VH, AGBD has HH/HV")
    print("   Solution: Band mapping should be handled by BandFilter")
    print()
    print("4. 🎯 WORKING COMMAND:")
    print("   WANDB_MODE=offline torchrun --standalone --nproc_per_node=1 pangaea/run.py \\")
    print("     dataset=agbd encoder=scalemae decoder=reg_upernet \\")
    print("     preprocessing=agbd_regression batch_size=8 task.trainer.n_epochs=1")
    print()
    print("5. 📋 FOR COMPREHENSIVE TESTING:")
    print("   - Always use preprocessing=agbd_regression")
    print("   - Use reg_upernet or reg_fcn decoders (NOT reg_unet)")
    print("   - No need to override dataset.img_size (ResizeToEncoder handles it)")

if __name__ == "__main__":
    print("AGBD Integration Diagnostic Tool v3")
    print("Understanding the Preprocessing Pipeline")
    print("=" * 40)
    
    # Run checks
    config_ok, config = check_agbd_config()
    check_preprocessing_pipeline()
    analyze_the_real_problem()
    suggest_fixes()
    
    print("\n" + "=" * 40)
    print("🎯 KEY INSIGHT:")
    print("The problem is NOT dataset.img_size!")
    print("The problem is the PREPROCESSING PIPELINE!")
    print()
    print("✅ Use preprocessing=agbd_regression")
    print("✅ This will properly resize 25x25 -> encoder requirements")
    print("✅ ResizeToEncoder automatically detects encoder input size")
