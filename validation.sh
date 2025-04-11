#!/usr/bin/env bash

# Colors for output (compatible with Git Bash)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Counters for tests
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=()

# Function to run a test
run_test() {
    local name=$1
    local command=$2
    
    echo -e "\n${CYAN}[TEST $((++TOTAL_TESTS))] $name${NC}"
    
    if eval "$command"; then
        echo -e "${GREEN}✓ PASSED${NC}"
        ((PASSED_TESTS++))
        return 0
    else
        echo -e "${RED}✗ FAILED${NC}"
        FAILED_TESTS+=("$name")
        return 1
    fi
}

echo -e "${CYAN}============================================"
echo "        SYSTEM VALIDATION SCRIPT"
echo -e "============================================${NC}"

# 1. Check .env file
run_test "Check .env file" '
if [ -f "./.env" ]; then
    echo "Found .env file"
    grep -q "AWS_FARGATE_ENDPOINT=" "./.env" && \
    grep -q "AZURE_DEEPSORT_ENDPOINT=" "./.env" && \
    echo "AWS Endpoint: $(grep "AWS_FARGATE_ENDPOINT=" "./.env")" && \
    echo "Azure Endpoint: $(grep "AZURE_DEEPSORT_ENDPOINT=" "./.env")"
else
    echo "Missing .env file"
    false
fi'

# 2. Check cost_config.ini
run_test "Check cost_config.ini" '
if [ -f "./ml/cost_config.ini" ]; then
    echo "Found cost_config.ini"
    grep -q "fargate_per_vcpu_hour" "./ml/cost_config.ini" && \
    grep -q "container_app_per_vcpu_hour" "./ml/cost_config.ini"
else
    echo "Missing cost_config.ini"
    false
fi'

# 3. Check Docker is running
run_test "Check Docker is running" '
docker info > /dev/null 2>&1'

# 4. Check Python installation
run_test "Check Python installation" '
python --version > /dev/null 2>&1'

# 5. Check core files exist
run_test "Check core Python files" '
files=("./ml/video_pipeline.py" "./ml/trackers.py" "./ml/detectors.py" "./ml/video_processor.py" 
       "./ml/cost_utils.py" "./ml/db_utils.py" "./ml/run_api.py")
missing=0
for file in "${files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Missing: $file"
        missing=1
    fi
done
[ $missing -eq 0 ]'

# 6. Check YOLO model
run_test "Check YOLOv8 model file" '[ -f "./ml/yolov8n.pt" ]'

# 7. Test YOLO detector initialization
run_test "Test YOLO detector" '
cd ml && python test_imports.py yolo && cd ..'

# 8. Test video pipeline initialization
run_test "Test video pipeline" '
cd ml && python test_imports.py pipeline && cd ..'

# 9. Test cloud storage initialization
run_test "Test cloud storage" '
cd ml && python test_imports.py storage && cd ..'

# 10. Test tracker initialization
run_test "Test tracker" '
cd ml && python test_imports.py tracker && cd ..'

# 11. Test MLflow initialization
run_test "Test MLflow" '
cd ml && python test_imports.py mlflow && cd ..'

# 12. Test database connection
run_test "Test database connection" '
cd ml && python test_imports.py database && cd ..'

# 13. Test API dependencies
run_test "Test API dependencies" '
cd ml && python test_imports.py api && cd ..'

# 14. Check cloud implementation files
run_test "Check cloud implementation files" '
[ -f "./cloud/aws/tracker_app.py" ] && \
[ -f "./cloud/azure/tracker_app.py" ]'

# Calculate pass rate
PASS_RATE=$(( (PASSED_TESTS * 100) / TOTAL_TESTS ))

# Print summary
echo -e "\n${CYAN}============================================"
echo "            VALIDATION RESULTS"
echo -e "============================================${NC}"
echo -e "Total Tests Run: ${TOTAL_TESTS}"
echo -e "Tests Passed:    ${GREEN}${PASSED_TESTS}${NC}"
echo -e "Tests Failed:    ${RED}$((TOTAL_TESTS - PASSED_TESTS))${NC}"

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo -e "\n${YELLOW}Failed Tests:${NC}"
    for test in "${FAILED_TESTS[@]}"; do
        echo -e "  - $test"
    done
fi

echo -e "\nOverall Pass Rate: ${PASS_RATE}%"

# Print final status
if [ $PASS_RATE -ge 80 ]; then
    echo -e "\n${GREEN}VALIDATION STATUS: SUCCESS${NC}"
    echo "System appears to be properly configured."
    echo -e "\n${CYAN}Next Steps:${NC}"
    echo "1. Run a complete end-to-end test through the GUI"
    echo "2. Upload a real video and verify all processors work correctly"
    echo "3. Check cost calculation accuracy"
elif [ $PASS_RATE -ge 60 ]; then
    echo -e "\n${YELLOW}VALIDATION STATUS: PARTIAL SUCCESS${NC}"
    echo "Some configuration issues need attention."
    echo -e "\n${CYAN}Recommended Actions:${NC}"
    echo "1. Address the failed tests above"
    echo "2. Check cloud endpoints and connectivity"
    echo "3. Verify environment variables and credentials"
else
    echo -e "\n${RED}VALIDATION STATUS: FAILED${NC}"
    echo "Significant configuration issues detected."
    echo -e "\n${CYAN}Required Actions:${NC}"
    echo "1. Ensure Docker is running"
    echo "2. Verify Python environment and dependencies"
    echo "3. Check all core files exist"
    echo "4. Configure AWS and Azure credentials"
fi

echo -e "\n${CYAN}============================================${NC}"