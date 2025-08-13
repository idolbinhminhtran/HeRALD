#!/bin/bash
# CI-style test runner for HeRALD project

echo "================================================================"
echo "HeRALD Test Suite"
echo "================================================================"

# Set error handling
set -e  # Exit on first error
set -u  # Exit on undefined variable

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Run indexing and sanity tests
echo -e "\n${GREEN}Running indexing and sanity tests...${NC}"
python3 tests/test_indexing.py
TEST_RESULT=$?

if [ $TEST_RESULT -eq 0 ]; then
    echo -e "\n${GREEN}✅ All tests passed successfully!${NC}"
    exit 0
else
    echo -e "\n${RED}❌ Some tests failed!${NC}"
    exit 1
fi
