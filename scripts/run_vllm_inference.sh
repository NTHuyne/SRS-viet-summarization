#!/bin/bash

# Script để chạy vLLM server và inference tự động
# Sử dụng: ./scripts/run_vllm_inference.sh --model-path /path/to/model [options]

set -e  # Exit on error

# Màu sắc cho output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
MODEL_PATH=""
MODEL_NAME="qwen-sft"
PORT=8000
TEMPERATURE=0.7
TOP_P=0.9
MAX_TOKENS=512
TEST_FILE="datasets-test/test.jsonl"
OUTPUT_DIR="results"
BATCH_SIZE=16
MAX_WORKERS=8
TENSOR_PARALLEL_SIZE=1
MAX_MODEL_LEN=4096

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
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
        --max-tokens)
            MAX_TOKENS="$2"
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
        --enable-lora)
            ENABLE_LORA=true
            shift
            ;;
        --lora-path)
            LORA_PATH="$2"
            shift 2
            ;;
        --tensor-parallel-size)
            TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --max-model-len)
            MAX_MODEL_LEN="$2"
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
        -h|--help)
            echo "Usage: $0 --model-path /path/to/model [options]"
            echo ""
            echo "Required:"
            echo "  --model-path PATH              Đường dẫn đến model"
            echo ""
            echo "Optional:"
            echo "  --model-name NAME              Tên model (default: qwen-sft)"
            echo "  --port PORT                    Port cho vLLM server (default: 8000)"
            echo "  --temperature TEMP             Temperature (default: 0.7)"
            echo "  --top-p VALUE                  Top-p (default: 0.9)"
            echo "  --max-tokens NUM               Max tokens (default: 512)"
            echo "  --test-file PATH               File test (default: datasets-test/test.jsonl)"
            echo "  --output-dir DIR               Output directory (default: results)"
            echo "  --batch-size NUM               Batch size cho inference (default: 16)"
            echo "  --max-workers NUM              Số threads song song (default: 8)"
            echo "  --tensor-parallel-size NUM     Số GPU (default: 1)"
            echo "  --max-model-len NUM            Max context length (default: 4096)"
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
echo -e "${GREEN}vLLM Inference Pipeline${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Model path: $MODEL_PATH"
echo "Model name: $MODEL_NAME"
echo "Port: $PORT"
echo "Temperature: $TEMPERATURE"
echo "Top-p: $TOP_P"
echo "Max tokens: $MAX_TOKENS"
echo "Test file: $TEST_FILE"
echo "Output dir: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Max workers: $MAX_WORKERS"
echo -e "${GREEN}========================================${NC}\n"

# Step 1: Start vLLM server in background
echo -e "${YELLOW}[1/3] Đang khởi động vLLM server...${NC}"

SERVER_CMD="python scripts/serve_vllm.py \
    --model-path $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --port $PORT \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --max-model-len $MAX_MODEL_LEN"

# Start server in background and save PID
$SERVER_CMD > vllm_server.log 2>&1 &
SERVER_PID=$!

echo "Server PID: $SERVER_PID"
echo "Server logs: vllm_server.log"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}Đang dừng vLLM server...${NC}"
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    echo -e "${GREEN}Server đã dừng${NC}"
}

trap cleanup EXIT INT TERM

# Step 2: Wait for server to be ready
echo -e "${YELLOW}[2/3] Đợi server khởi động...${NC}"

MAX_WAIT=300  # 5 minutes
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Server đã sẵn sàng!${NC}\n"
        break
    fi
    
    # Check if server process is still running
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo -e "${RED}Error: Server process died. Check vllm_server.log for details${NC}"
        tail -n 20 vllm_server.log
        exit 1
    fi
    
    sleep 5
    WAITED=$((WAITED + 5))
    echo "Đã đợi ${WAITED}s..."
done

if [ $WAITED -ge $MAX_WAIT ]; then
    echo -e "${RED}Error: Server không khởi động sau $MAX_WAIT giây${NC}"
    echo "Check vllm_server.log for details"
    exit 1
fi

# Step 3: Run inference
echo -e "${YELLOW}[3/3] Đang chạy inference...${NC}\n"

python scripts/inference_test.py \
    --model-name "$MODEL_NAME" \
    --api-url "http://localhost:$PORT/v1" \
    --test-file "$TEST_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --max-tokens "$MAX_TOKENS" \
    --batch-size "$BATCH_SIZE" \
    --max-workers "$MAX_WORKERS"

INFERENCE_STATUS=$?

if [ $INFERENCE_STATUS -eq 0 ]; then
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}✓ Inference hoàn thành thành công!${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    # Show output file
    OUTPUT_FILE=$(ls -t "$OUTPUT_DIR"/${MODEL_NAME}_*.jsonl 2>/dev/null | head -n 1)
    if [ -n "$OUTPUT_FILE" ]; then
        echo "Kết quả: $OUTPUT_FILE"
        echo "Số dòng: $(wc -l < "$OUTPUT_FILE")"
    fi
else
    echo -e "\n${RED}========================================${NC}"
    echo -e "${RED}✗ Inference thất bại${NC}"
    echo -e "${RED}========================================${NC}"
    exit $INFERENCE_STATUS
fi
