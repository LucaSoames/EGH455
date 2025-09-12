#!/usr/bin/env python3
"""
Chrony NTP Synchronization Checker and Fixer

This script checks if chrony NTP service is running and properly synchronized.
If issues are detected, it attempts to fix them automatically.

Features:
- Checks if chrony service is running
- Verifies NTP synchronization status
- Starts chrony service if stopped
- Forces synchronization if needed
- Reloads NTP sources for network changes

Usage:
    python3 check_chrony.py
    # or make executable and run directly:
    ./check_chrony.py

Exit codes:
    0: Success - chrony is running and synchronized
    1: Error - chrony issues detected and could not be fixed
"""

import subprocess
import sys
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_command(cmd, capture_output=True):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=capture_output,
            text=True,
            timeout=30
        )
        stdout = result.stdout.strip() if result.stdout else ""
        stderr = result.stderr.strip() if result.stderr else ""
        return result.returncode, stdout, stderr
    except subprocess.TimeoutExpired:
        logging.error(f"Command timed out: {cmd}")
        return -1, "", "Command timed out"
    except Exception as e:
        logging.error(f"Error running command '{cmd}': {e}")
        return -1, "", str(e)

def check_chrony_service():
    """Check if chrony service is running."""
    logging.info("Checking chrony service status...")
    returncode, stdout, stderr = run_command("sudo systemctl is-active chrony")

    if returncode == 0 and stdout == "active":
        logging.info("✓ Chrony service is running")
        return True
    else:
        logging.warning("✗ Chrony service is not running")
        return False

def start_chrony_service():
    """Start the chrony service."""
    logging.info("Attempting to start chrony service...")

    # Enable chrony to start on boot
    returncode, stdout, stderr = run_command("sudo systemctl enable chrony")
    if returncode != 0:
        logging.error(f"Failed to enable chrony: {stderr}")
        return False

    # Start the service
    returncode, stdout, stderr = run_command("sudo systemctl start chrony")
    if returncode != 0:
        logging.error(f"Failed to start chrony: {stderr}")
        return False

    # Wait a moment for service to start
    time.sleep(2)

    # Verify it's now running
    if check_chrony_service():
        logging.info("✓ Chrony service started successfully")
        return True
    else:
        logging.error("✗ Failed to start chrony service")
        return False

def check_chrony_sync():
    """Check if chrony is synchronized."""
    logging.info("Checking chrony synchronization status...")

    returncode, stdout, stderr = run_command("chronyc tracking")
    if returncode != 0:
        logging.error(f"Failed to get tracking info: {stderr}")
        return False

    # Check for key indicators of synchronization
    lines = stdout.split('\n')
    stratum = None
    system_time_offset = None
    leap_status = None

    for line in lines:
        if line.startswith('Stratum'):
            try:
                stratum = int(line.split(':')[1].strip())
            except (ValueError, IndexError):
                pass
        elif line.startswith('System time'):
            # Extract the offset value
            try:
                offset_str = line.split(':')[1].strip().split()[0]
                system_time_offset = float(offset_str)
            except (ValueError, IndexError):
                pass
        elif line.startswith('Leap status'):
            leap_status = line.split(':')[1].strip()

    # Evaluate synchronization status
    if stratum is None:
        logging.warning("Could not determine stratum")
        return False

    if stratum == 0:
        logging.warning("Stratum is 0 - not synchronized")
        return False

    if system_time_offset is None:
        logging.warning("Could not determine system time offset")
        return False

    # Check if offset is reasonable (within 1 second)
    if abs(system_time_offset) > 1.0:
        logging.warning(f"Large time offset detected: {system_time_offset} seconds")
        return False

    if leap_status and leap_status not in ['Normal', 'Insert second', 'Delete second']:
        logging.warning(f"Unusual leap status: {leap_status}")
        return False

    logging.info(f"✓ Chrony synchronized - Stratum: {stratum}, Offset: {system_time_offset:.6f}s")
    return True

def force_chrony_sync():
    """Force chrony to synchronize immediately."""
    logging.info("Attempting to force time synchronization...")

    # Try makestep first
    returncode, stdout, stderr = run_command("sudo chronyc makestep")
    if returncode == 0:
        logging.info("✓ Makestep completed")
    else:
        logging.warning(f"Makestep failed: {stderr}")

    # Wait a moment
    time.sleep(2)

    # Reload sources in case network changed
    returncode, stdout, stderr = run_command("sudo chronyc reload sources")
    if returncode == 0:
        logging.info("✓ Sources reloaded")
    else:
        logging.warning(f"Failed to reload sources: {stderr}")

    # Wait for sync
    time.sleep(5)

    return check_chrony_sync()

def check_ntp_sources():
    """Check if NTP sources are available."""
    logging.info("Checking NTP sources...")

    returncode, stdout, stderr = run_command("chronyc sources")
    if returncode != 0:
        logging.error(f"Failed to get sources: {stderr}")
        return False

    lines = stdout.split('\n')
    source_count = 0

    for line in lines:
        if line.startswith('^') or line.startswith('*') or line.startswith('+') or line.startswith('-'):
            source_count += 1

    if source_count == 0:
        logging.warning("No NTP sources found")
        return False

    logging.info(f"✓ Found {source_count} NTP sources")
    return True

def main():
    """Main function to check and fix chrony."""
    print("Chrony NTP Synchronization Checker")
    print("=" * 40)

    issues_found = False

    # Check if chrony is installed
    returncode, stdout, stderr = run_command("dpkg -l chrony")
    if returncode != 0:
        logging.error("Chrony is not installed!")
        print("Please install chrony first: sudo apt install chrony")
        sys.exit(1)

    # Check service status
    if not check_chrony_service():
        issues_found = True
        if not start_chrony_service():
            logging.error("Cannot proceed without chrony service")
            sys.exit(1)

    # Check sources
    if not check_ntp_sources():
        issues_found = True
        logging.warning("NTP sources may be unavailable")

    # Check synchronization
    if not check_chrony_sync():
        issues_found = True
        logging.warning("Time synchronization issues detected")

        # Try to fix sync issues
        if force_chrony_sync():
            logging.info("✓ Synchronization fixed")
        else:
            logging.error("✗ Could not fix synchronization issues")
            print("\nTroubleshooting suggestions:")
            print("- Check network connectivity")
            print("- Verify NTP sources are accessible")
            print("- Check firewall settings")
            print("- Try restarting the Pi")
            sys.exit(1)

    if not issues_found:
        print("\n✓ All checks passed - Chrony is running and synchronized")
    else:
        print("\n✓ Issues detected and resolved")

    print("\nCurrent status:")
    run_command("chronyc tracking", capture_output=False)

if __name__ == "__main__":
    main()