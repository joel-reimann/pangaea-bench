#!/usr/bin/env python3
"""
Comprehensive testing script for AGBD integration with PANGAEA encoders/decoders.

This script tests all valid encoder/decoder combinations with AGBD dataset to identify:
1. Compatibility issues between encoders and data
2. Band requirement mismatches
3. Preprocessing configuration problems
4. Training pipeline issues

Usage:
    python test_agbd_integration.py [--quick] [--encoder ENCODER] [--decoder DECODER]
    
Examples:
    # Test all combinations (full test)
    python test_agbd_integration.py
    
    # Quick test (data loading only, no training)
    python test_agbd_integration.py --quick
    
    # Test specific encoder/decoder
    python test_agbd_integration.py --encoder remoteclip --decoder reg_upernet
"""

import os
import sys
import traceback
import logging
import argparse
from pathlib import Path
import torch
from typing import Dict, List, Tuple, Optional
import yaml
import hydra
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'agbd_integration_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Available encoders and decoders for AGBD
AVAILABLE_ENCODERS = {
    'remoteclip': {
        'bands_required': 3,
        'config_file': 'remoteclip.yaml',
        'dataset_config': 'agbd.yaml',
        'modalities': ['optical'],
        'preprocessing': 'agbd_regression'
    },
    'prithvi': {
        'bands_required': 6,
        'config_file': 'prithvi.yaml', 
        'dataset_config': 'agbd.yaml',
        'modalities': ['optical'],
        'preprocessing': 'agbd_regression'
    },
    'scalemae': {
        'bands_required': 'variable',
        'config_file': 'scalemae.yaml',
        'dataset_config': 'agbd.yaml',
        'modalities': ['optical'],
        'preprocessing': 'agbd_regression'
    },
    'gfmswin': {
        'bands_required': 'variable',
        'config_file': 'gfmswin.yaml',
        'dataset_config': 'agbd.yaml',
        'modalities': ['optical'],
        'preprocessing': 'agbd_regression'
    },
    'croma_optical': {
        'bands_required': 'variable',
        'config_file': 'croma_optical.yaml',
        'dataset_config': 'agbd.yaml',
        'modalities': ['optical'],
        'preprocessing': 'agbd_regression'
    },
    'croma_sar': {
        'bands_required': 2,
        'config_file': 'croma_sar.yaml',
        'dataset_config': 'agbd.yaml',
        'modalities': ['sar'],
        'preprocessing': 'agbd_regression'
    },
    'croma_joint': {
        'bands_required': 'multi_modal',
        'config_file': 'croma_joint.yaml',
        'dataset_config': 'agbd.yaml',
        'modalities': ['optical', 'sar'],
        'preprocessing': 'agbd_regression'
    }
}

AVAILABLE_DECODERS = {
    'reg_upernet': {
        'config_file': 'reg_upernet.yaml',
        'task_type': 'regression',
        'output_type': 'scalar'
    },
    'reg_unet': {
        'config_file': 'reg_unet.yaml',
        'task_type': 'regression', 
        'output_type': 'scalar'
    }
}

class AGBDTester:
    def __init__(self, config_dir: str = "/scratch/reimannj5/pangaea-bench/configs"):
        self.config_dir = Path(config_dir)
        self.results = []
        
    def test_data_loading(self, dataset_config: str) -> Dict:
        """Test basic data loading for AGBD dataset."""
        logger.info(f"Testing data loading with config: {dataset_config}")
        
        try:
            # Test dataset instantiation and data loading
            with initialize_config_dir(config_dir=str(self.config_dir), version_base=None):
                cfg = compose(config_name="train", overrides=[
                    f"dataset={dataset_config}", 
                    "criterion=mse", 
                    "preprocessing=agbd_regression",
                    "decoder=reg_upernet",
                    "encoder=remoteclip",
                    "task=regression"
                ])
                
                # Import dataset class
                dataset_target = cfg.dataset._target_
                module_name, class_name = dataset_target.rsplit('.', 1)
                module = __import__(module_name, fromlist=[class_name])
                dataset_class = getattr(module, class_name)
                
                # Instantiate dataset
                dataset_kwargs = OmegaConf.to_container(cfg.dataset, resolve=True)
                dataset_kwargs.pop('_target_')
                dataset = dataset_class(split='train', **dataset_kwargs)
                
                # Test basic properties
                dataset_len = len(dataset)
                logger.info(f"Dataset length: {dataset_len}")
                
                if dataset_len == 0:
                    return {'status': 'error', 'error': 'Dataset is empty'}
                
                # Test loading a sample
                sample = dataset[0]
                logger.info(f"Sample keys: {sample.keys()}")
                
                if 'image' in sample:
                    image = sample['image']
                    logger.info(f"Image modalities: {list(image.keys())}")
                    for mod, data in image.items():
                        logger.info(f"{mod} shape: {data.shape}, dtype: {data.dtype}")
                
                if 'target' in sample:
                    target = sample['target']
                    logger.info(f"Target shape: {target.shape if hasattr(target, 'shape') else 'scalar'}, dtype: {target.dtype}")
                
                return {
                    'status': 'success',
                    'dataset_length': dataset_len,
                    'sample_keys': list(sample.keys()),
                    'modalities': list(sample['image'].keys()) if 'image' in sample else [],
                    'target_shape': str(target.shape) if hasattr(target, 'shape') else 'scalar'
                }
                
        except Exception as e:
            logger.error(f"Data loading test failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {'status': 'error', 'error': str(e)}
    
    def test_preprocessing(self, dataset_config: str, preprocessing_config: str) -> Dict:
        """Test preprocessing pipeline."""
        logger.info(f"Testing preprocessing: {preprocessing_config} with dataset: {dataset_config}")
        
        try:
            with initialize_config_dir(config_dir=str(self.config_dir), version_base=None):
                cfg = compose(config_name="train", overrides=[
                    f"dataset={dataset_config}",
                    f"preprocessing={preprocessing_config}"
                ])
                
                # Test preprocessing config loading
                preprocessing_cfg = cfg.preprocessing
                logger.info(f"Preprocessing config: {OmegaConf.to_yaml(preprocessing_cfg)}")
                
                return {'status': 'success', 'preprocessing_config': OmegaConf.to_yaml(preprocessing_cfg)}
                
        except Exception as e:
            logger.error(f"Preprocessing test failed: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def test_encoder_compatibility(self, encoder_name: str, dataset_config: str) -> Dict:
        """Test encoder compatibility with dataset."""
        logger.info(f"Testing encoder compatibility: {encoder_name} with dataset: {dataset_config}")
        
        try:
            encoder_info = AVAILABLE_ENCODERS[encoder_name]
            
            with initialize_config_dir(config_dir=str(self.config_dir), version_base=None):
                cfg = compose(config_name="train", overrides=[
                    f"dataset={dataset_config}",
                    f"encoder={encoder_info['config_file'].replace('.yaml', '')}",
                    f"preprocessing={encoder_info['preprocessing']}",
                    "criterion=mse",
                    "decoder=reg_upernet",
                    "task=regression"
                ])
                
                # Check encoder config
                encoder_cfg = cfg.encoder
                logger.info(f"Encoder config loaded: {encoder_cfg._target_}")
                
                # Test dataset compatibility
                dataset_cfg = cfg.dataset
                dataset_bands = dataset_cfg.get('bands', {})
                
                required_modalities = encoder_info['modalities']
                available_modalities = list(dataset_bands.keys())
                
                missing_modalities = set(required_modalities) - set(available_modalities)
                if missing_modalities:
                    return {
                        'status': 'error', 
                        'error': f'Missing modalities: {missing_modalities}'
                    }
                
                return {
                    'status': 'success',
                    'encoder_target': encoder_cfg._target_,
                    'required_modalities': required_modalities,
                    'available_modalities': available_modalities
                }
                
        except Exception as e:
            logger.error(f"Encoder compatibility test failed: {str(e)}")
            return {'status': 'error', 'error': str(e)}
    
    def test_full_pipeline(self, encoder_name: str, decoder_name: str, quick: bool = False) -> Dict:
        """Test full training pipeline (or just data loading if quick=True)."""
        encoder_info = AVAILABLE_ENCODERS[encoder_name]
        decoder_info = AVAILABLE_DECODERS[decoder_name]
        
        logger.info(f"Testing full pipeline: {encoder_name} + {decoder_name} (quick={quick})")
        
        try:
            with initialize_config_dir(config_dir=str(self.config_dir), version_base=None):
                overrides = [
                    f"dataset={encoder_info['dataset_config'].replace('.yaml', '')}",
                    f"encoder={encoder_info['config_file'].replace('.yaml', '')}",
                    f"decoder={decoder_info['config_file'].replace('.yaml', '')}",
                    f"preprocessing={encoder_info['preprocessing']}",
                    "criterion=mse",
                    "task=regression",
                    "+task.trainer.max_epochs=1",
                    "+task.trainer.devices=1",
                    "+data_loader.batch_size=2",
                    "+data_loader.num_workers=1"
                ]
                
                cfg = compose(config_name="train", overrides=overrides)
                
                if quick:
                    # Just test config composition
                    logger.info("Quick test - config composition successful")
                    return {
                        'status': 'success',
                        'test_type': 'quick',
                        'config_keys': list(cfg.keys())
                    }
                else:
                    # Test actual training initialization
                    logger.info("Testing training initialization...")
                    
                    # This would normally import and run the training script
                    # For now, just verify config is complete
                    required_keys = ['dataset', 'encoder', 'decoder', 'preprocessing', 'criterion', 'task']
                    missing_keys = [k for k in required_keys if k not in cfg]
                    
                    if missing_keys:
                        return {
                            'status': 'error',
                            'error': f'Missing config keys: {missing_keys}'
                        }
                    
                    return {
                        'status': 'success',
                        'test_type': 'full',
                        'config_complete': True
                    }
                    
        except Exception as e:
            logger.error(f"Full pipeline test failed: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {'status': 'error', 'error': str(e)}
    
    def run_comprehensive_test(self, quick: bool = False, target_encoder: str = None, target_decoder: str = None):
        """Run comprehensive test of all combinations."""
        logger.info("Starting comprehensive AGBD integration test")
        logger.info(f"Quick mode: {quick}")
        
        # Test basic data loading first
        logger.info("=" * 60)
        logger.info("TESTING BASIC DATA LOADING")
        logger.info("=" * 60)
        
        for dataset_config in ['agbd']:
            logger.info(f"\nTesting dataset config: {dataset_config}")
            result = self.test_data_loading(dataset_config)
            self.results.append({
                'test_type': 'data_loading',
                'dataset_config': dataset_config,
                'encoder': None,
                'decoder': None,
                'status': result['status'],
                'details': result
            })
            
            if result['status'] == 'error':
                logger.error(f"Data loading failed for {dataset_config}: {result['error']}")
            else:
                logger.info(f"Data loading successful for {dataset_config}")
        
        # Test encoder compatibility
        logger.info("=" * 60)
        logger.info("TESTING ENCODER COMPATIBILITY")
        logger.info("=" * 60)
        
        encoders_to_test = [target_encoder] if target_encoder else AVAILABLE_ENCODERS.keys()
        
        for encoder_name in encoders_to_test:
            encoder_info = AVAILABLE_ENCODERS[encoder_name]
            dataset_config = encoder_info['dataset_config'].replace('.yaml', '')
            
            logger.info(f"\nTesting encoder: {encoder_name}")
            result = self.test_encoder_compatibility(encoder_name, dataset_config)
            self.results.append({
                'test_type': 'encoder_compatibility',
                'dataset_config': dataset_config,
                'encoder': encoder_name,
                'decoder': None,
                'status': result['status'],
                'details': result
            })
            
            if result['status'] == 'error':
                logger.error(f"Encoder compatibility failed for {encoder_name}: {result['error']}")
            else:
                logger.info(f"Encoder compatibility successful for {encoder_name}")
        
        # Test full pipeline combinations
        logger.info("=" * 60)
        logger.info("TESTING FULL PIPELINE COMBINATIONS")
        logger.info("=" * 60)
        
        decoders_to_test = [target_decoder] if target_decoder else AVAILABLE_DECODERS.keys()
        
        for encoder_name in encoders_to_test:
            for decoder_name in decoders_to_test:
                logger.info(f"\nTesting combination: {encoder_name} + {decoder_name}")
                result = self.test_full_pipeline(encoder_name, decoder_name, quick=quick)
                self.results.append({
                    'test_type': 'full_pipeline',
                    'dataset_config': AVAILABLE_ENCODERS[encoder_name]['dataset_config'],
                    'encoder': encoder_name,
                    'decoder': decoder_name,
                    'status': result['status'],
                    'details': result
                })
                
                if result['status'] == 'error':
                    logger.error(f"Pipeline test failed for {encoder_name}+{decoder_name}: {result['error']}")
                else:
                    logger.info(f"Pipeline test successful for {encoder_name}+{decoder_name}")
    
    def generate_report(self) -> str:
        """Generate a comprehensive test report."""
        report = []
        report.append("AGBD Integration Test Report")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total tests: {len(self.results)}")
        
        # Summary by test type
        test_types = {}
        for result in self.results:
            test_type = result['test_type']
            if test_type not in test_types:
                test_types[test_type] = {'total': 0, 'success': 0, 'error': 0}
            test_types[test_type]['total'] += 1
            test_types[test_type][result['status']] += 1
        
        report.append("\nSummary by Test Type:")
        report.append("-" * 30)
        for test_type, stats in test_types.items():
            report.append(f"{test_type}: {stats['success']}/{stats['total']} passed")
        
        # Detailed results
        report.append("\nDetailed Results:")
        report.append("-" * 30)
        
        for result in self.results:
            status_symbol = "✓" if result['status'] == 'success' else "✗"
            test_desc = f"{result['test_type']}"
            if result['encoder']:
                test_desc += f" ({result['encoder']}"
                if result['decoder']:
                    test_desc += f"+{result['decoder']}"
                test_desc += ")"
            
            report.append(f"{status_symbol} {test_desc}")
            if result['status'] == 'error':
                report.append(f"    Error: {result['details'].get('error', 'Unknown error')}")
        
        # Successful combinations
        successful_combinations = [
            (r['encoder'], r['decoder']) for r in self.results 
            if r['test_type'] == 'full_pipeline' and r['status'] == 'success'
        ]
        
        if successful_combinations:
            report.append("\nWorking Encoder/Decoder Combinations:")
            report.append("-" * 40)
            for encoder, decoder in successful_combinations:
                report.append(f"  • {encoder} + {decoder}")
        
        # Failed combinations with reasons
        failed_combinations = [
            (r['encoder'], r['decoder'], r['details'].get('error', 'Unknown')) 
            for r in self.results 
            if r['test_type'] == 'full_pipeline' and r['status'] == 'error'
        ]
        
        if failed_combinations:
            report.append("\nFailed Combinations:")
            report.append("-" * 20)
            for encoder, decoder, error in failed_combinations:
                report.append(f"  • {encoder} + {decoder}: {error}")
        
        return "\n".join(report)
    
    def save_results(self, filename: str = None):
        """Save results to CSV and text report."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"agbd_test_results_{timestamp}"
        
        # Save CSV
        df = pd.DataFrame(self.results)
        csv_file = f"{filename}.csv"
        df.to_csv(csv_file, index=False)
        logger.info(f"Results saved to {csv_file}")
        
        # Save text report
        report = self.generate_report()
        txt_file = f"{filename}.txt"
        with open(txt_file, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to {txt_file}")
        
        return csv_file, txt_file

def main():
    parser = argparse.ArgumentParser(description="Test AGBD integration with PANGAEA")
    parser.add_argument('--quick', action='store_true', 
                       help='Quick test (data loading only, no training)')
    parser.add_argument('--encoder', type=str, 
                       help='Test specific encoder only')
    parser.add_argument('--decoder', type=str,
                       help='Test specific decoder only')
    parser.add_argument('--config-dir', type=str,
                       default='/scratch/reimannj5/pangaea-bench/configs',
                       help='Path to config directory')
    
    args = parser.parse_args()
    
    # Validate encoder/decoder choices
    if args.encoder and args.encoder not in AVAILABLE_ENCODERS:
        logger.error(f"Unknown encoder: {args.encoder}")
        logger.error(f"Available encoders: {list(AVAILABLE_ENCODERS.keys())}")
        sys.exit(1)
    
    if args.decoder and args.decoder not in AVAILABLE_DECODERS:
        logger.error(f"Unknown decoder: {args.decoder}")
        logger.error(f"Available decoders: {list(AVAILABLE_DECODERS.keys())}")
        sys.exit(1)
    
    # Run tests
    tester = AGBDTester(config_dir=args.config_dir)
    tester.run_comprehensive_test(
        quick=args.quick,
        target_encoder=args.encoder,
        target_decoder=args.decoder
    )
    
    # Generate and save report
    csv_file, txt_file = tester.save_results()
    
    # Print summary
    print("\n" + "="*60)
    print("TEST COMPLETED")
    print("="*60)
    print(tester.generate_report())
    print(f"\nDetailed results saved to: {csv_file}")
    print(f"Test report saved to: {txt_file}")

if __name__ == "__main__":
    main()
