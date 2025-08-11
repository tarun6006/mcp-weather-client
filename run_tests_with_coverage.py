#!/usr/bin/env python3
"""
Test runner script for Client (Slack Bot) with coverage reporting
Enforces 90% code coverage requirement
"""
import subprocess
import sys
import os
import json
import logging
from datetime import datetime

# Setup logging for test runner
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def main():
    """Run tests with coverage for Client (Slack Bot)"""
    logger.info("Client (Slack Bot) - Test Suite with Coverage")
    logger.info("Running black box tests with 90% coverage requirement")
    logger.info("="*60)
    
    try:
        # Run pytest with coverage (using configuration from pytest.ini)
        result = subprocess.run([
            sys.executable, "-m", "pytest"
        ], capture_output=True, text=True)
        
        logger.info("Test output:")
        logger.info(f"{result.stdout}")
        if result.stderr:
            logger.error(f"STDERR: {result.stderr}")
        
        # Try to read coverage report
        coverage_data = None
        if os.path.exists("coverage.json"):
            try:
                with open("coverage.json", 'r') as f:
                    coverage_data = json.load(f)
                    coverage_percent = coverage_data.get('totals', {}).get('percent_covered', 0)
                    logger.info(f"Coverage: {coverage_percent:.2f}%")
                    
                    # Create summary report
                    summary_report = {
                        "timestamp": datetime.now().isoformat(),
                        "application": "Client (Slack Bot)",
                        "tests_passed": result.returncode == 0,
                        "coverage_percentage": coverage_percent,
                        "coverage_meets_threshold": coverage_percent >= 90,
                        "coverage_data": coverage_data
                    }
                    
                    # Save summary
                    with open("test_summary.json", "w") as f:
                        json.dump(summary_report, f, indent=2)
                    
            except Exception as e:
                logger.warning(f"Could not read coverage report: {e}")
        
        if result.returncode == 0:
            logger.info("Client (Slack Bot) tests passed with sufficient coverage!")
            sys.exit(0)
        else:
            logger.error("Client (Slack Bot) tests failed or coverage below 90%%!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

