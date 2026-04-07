#!/bin/bash

# Script để chạy inference với Transformers
# Sử dụng: ./scripts/run_inference_transformers.sh --model-path /path/to/model [options]

set -e  # Exit on error

# Màu sắc cho output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
MODEL_PATH=""
TEST_FILE="datasets-test/test.jsonl"
OUTPUT_DIR="results"
OUTPUT_NAME=""
TEMPERATURE=0.7
TOP_P=0.9
MAX_NEW_TOKENS=512
BATCH_SIZE=16
MAX_WORKERS=4
DEVICE="auto"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --test-file)
            TEST_FILE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --output-name)
            OUTPUT_NAME="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --top-p)
            TOP_P="$2"
            shift 2
            ;;
        --max-new-tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 --model-path /path/to/model [options]"
            echo ""
            echo "Required:"
            echo "  --model-path PATH              Đường dẫn đến model"
            echo ""
            echo "Optional:"
            echo "  --test-file PATH               File test (default: datasets-test/test.jsonl)"
            echo "  --output-dir DIR               Output directory (default: results)"
            echo "  --output-name NAME             Tên file output (không có .csv)"
            echo "  --temperature TEMP             Temperature (default: 0.7)"
            echo "  --top-p VALUE                  Top-p (default: 0.9)"
            echo "  --max-new-tokens NUM           Max new tokens (default: 512)"
            echo "  --batch-size NUM               Batch size (default: 16)"
            echo "  --max-workers NUM              Số threads song song (default: 4)"
            echo "  --device DEVICE                Device: auto, cuda, cpu (default: auto)"
            echo "  -h, --help                     Hiển thị help"
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$MODEL_PATH" ]; then
    echo -e "${RED}Error: --model-path is required${NC}"
    echo "Use --help for usage information"
    exit 1
fi

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model path does not exist: $MODEL_PATH${NC}"
    exit 1
fi

# Check if test file exists
if [ ! -f "$TEST_FILE" ]; then
    echo -e "${RED}Error: Test file does not exist: $TEST_FILE${NC}"
    exit 1
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Transformers Inference Pipeline${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Model path: $MODEL_PATH"
echo "Test file: $TEST_FILE"
echo "Output dir: $OUTPUT_DIR"
if [ -n "$OUTPUT_NAME" ]; then
    echo "Output name: $OUTPUT_NAME.csv"
fi
echo "Temperature: $TEMPERATURE"
echo "Top-p: $TOP_P"
echo "Max new tokens: $MAX_NEW_TOKENS"
echo "Batch size: $BATCH_SIZE"
echo "Max workers: $MAX_WORKERS"
echo "Device: $DEVICE"
echo -e "${GREEN}========================================${NC}\n"

# Build command
CMD="python scripts/inference_test_transformers.py \
    --model-path \"$MODEL_PATH\" \
    --test-file \"$TEST_FILE\" \
    --output-dir \"$OUTPUT_DIR\" \
    --temperature $TEMPERATURE \
    --top-p $TOP_P \
    --max-new-tokens $MAX_NEW_TOKENS \
    --batch-size $BATCH_SIZE \
    --max-workers $MAX_WORKERS \
    --device $DEVICE"

if [ -n "$OUTPUT_NAME" ]; then
    CMD="$CMD --output-name \"$OUTPUT_NAME\""
fi

echo -e "${YELLOW}Đang chạy inference...${NC}\n"

# Run inference
eval $CMD

INFERENCE_STATUS=$?

if [ $INFERENCE_STATUS -eq 0 ]; then
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ Inference hoàn thành thành công!${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    # Show output file
    if [ -n "$OUTPUT_NAME" ]; then
        OUTPUT_FILE="$OUTPUT_DIR/${OUTPUT_NAME}.csv"
    else
        OUTPUT_FILE=$(ls -t "$OUTPUT_DIR"/*.csv 2>/dev/null | head -n 1)
    fi
    
    if [ -n "$OUTPUT_FILE" ] && [ -f "$OUTPUT_FILE" ]; then
        echo "Kết quả: $OUTPUT_FILE"
        echo "Số dòng: $(wc -l < "$OUTPUT_FILE")"
    fi
else
    echo -e "\n${RED}========================================${NC}"
    echo -e "${RED}✗ Inference thất bại${NC}"
    echo -e "${RED}========================================${NC}"
    exit $INFERENCE_STATUS
fi
