#!/usr/bin/env python3
"""
TAIP Subsystem - Installation and Testing Script
This script helps validate the installation and test the TAIP subsystem components.
"""

import sys
import subprocess
from pathlib import Path

def check_file_exists(filepath):
    """Check if a file exists and show its size."""
    path = Path(filepath)
    if path.exists():
        size = path.stat().st_size
        print(f"✓ {filepath} ({size:,} bytes)")
        return True
    else:
        print(f"✗ {filepath} (missing)")
        return False

def main():
    print("="*60)
    print("TAIP Subsystem Installation Verification")
    print("EGH455 UAVPayloadTAQ Project")
    print("="*60)
    
    # Check all generated files
    files_to_check = [
        "config.py",
        "data_models.py", 
        "vision_processing.py",
        "oak_camera.py",
        "gcs_client.py",
        "main.py"
    ]
    
    print("\n1. Checking generated files:")
    all_files_exist = True
    for filename in files_to_check:
        if not check_file_exists(filename):
            all_files_exist = False
    
    if not all_files_exist:
        print("\n✗ Some files are missing. Please regenerate them.")
        return False
    
    print("\n2. Testing imports:")
    test_imports = [
        ("config", "Configuration module"),
        ("data_models", "Data models"),
        ("vision_processing", "Vision processing"),
        ("oak_camera", "OAK camera interface"),
        ("gcs_client", "GCS client"),
    ]
    
    import_success = True
    for module, description in test_imports:
        try:
            __import__(module)
            print(f"✓ {description}")
        except ImportError as e:
            print(f"✗ {description}: {e}")
            import_success = False
        except Exception as e:
            print(f"⚠ {description}: Warning - {e}")
    
    print("\n3. Testing configuration validation:")
    try:
        import config
        config.validate_config()
        print("✓ Configuration validation passed")
    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        import_success = False
    
    print("\n4. Testing data models:")
    try:
        from data_models import create_test_payload
        test_payload = create_test_payload()
        test_payload.validate()
        print("✓ Data models working correctly")
        print(f"  - Test payload created with {len(test_payload.detections)} detections")
    except Exception as e:
        print(f"✗ Data models test failed: {e}")
        import_success = False
    
    print("\n5. Testing vision processing:")
    try:
        from vision_processing import validate_gauge_calibration
        if validate_gauge_calibration():
            print("✓ Vision processing calibration valid")
        else:
            print("⚠ Vision processing calibration issues detected")
    except Exception as e:
        print(f"✗ Vision processing test failed: {e}")
    
    print("\n" + "="*60)
    if all_files_exist and import_success:
        print("✓ TAIP Subsystem installation verification PASSED")
        print("\nNext steps:")
        print("1. Ensure OAK-D Lite camera is connected")
        print("2. Ensure Pimoroni Enviro+ HAT is connected")
        print("3. Configure GCS_BASE_URL in config.py")
        print("4. Run: python3 main.py")
    else:
        print("✗ TAIP Subsystem installation verification FAILED")
        print("Please check the errors above and fix them before running the system.")
    
    print("="*60)
    return all_files_exist and import_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
