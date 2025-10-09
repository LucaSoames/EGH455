#!/bin/bash
# Quick test script for audit logging system

echo "=========================================="
echo "  AUDIT LOGGING SYSTEM QUICK TEST"
echo "=========================================="
echo ""

# Check if database exists
if [ -f "/home/pi/EGH455/TAIP/audit_logs.db" ]; then
    echo "✓ Database exists"
else
    echo "✗ Database not found (will be created on first use)"
fi

# Test audit logger
echo ""
echo "Testing audit logger module..."
cd /home/pi/EGH455/TAIP
python3 test_audit_logging.py

# Show summary
echo ""
echo "=========================================="
echo "  TEST COMPLETE"
echo "=========================================="
echo ""
echo "To view logs in web interface:"
echo "  1. Start server: python3 TAIP/gcs_server.py"
echo "  2. Open: http://localhost:3000"
echo "  3. Click 'Audit Logs (Database)' tab"
echo ""
echo "To test with live data:"
echo "  python3 TAIP/main.py"
echo ""
echo "To send mock telemetry:"
echo "  python3 TAIP/gcs_client.py"
echo ""
