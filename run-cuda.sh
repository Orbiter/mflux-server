#!/bin/bash

# mflux-server CUDA control script
# Usage: ./run-cuda.sh [start|stop|restart|status] [--server-args]

# Change to the directory where the script is located
cd "$(dirname "$0")"

# Configuration
PROJECT_DIR="$(pwd)"
PID_FILE="$PROJECT_DIR/.mflux-server-cuda.pid"
LOG_FILE="$PROJECT_DIR/mflux-server-cuda.log"
DEFAULT_HOST="0.0.0.0"
DEFAULT_PORT="4030"
DEFAULT_MODEL="schnell"

# Ensure uv is installed
ensure_uv() {
    if ! command -v uv &> /dev/null; then
        echo "uv is not installed. Installing..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.cargo/bin:$PATH"
    fi
}

# Get the server process ID from PID file
get_pid() {
    if [ -f "$PID_FILE" ]; then
        cat "$PID_FILE"
    fi
}

# Check if server is running
is_running() {
    local pid=$(get_pid)
    if [ -n "$pid" ]; then
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

# Start the server
start_server() {
    if is_running; then
        echo "Server is already running (PID: $(get_pid))"
        return 1
    fi

    echo "Starting mflux-server (CUDA)..."

    ensure_uv

    # Sync dependencies with uv
    echo "Syncing dependencies..."
    uv sync --extra cuda --prerelease=allow

    # Parse additional arguments
    local args="--model $DEFAULT_MODEL"
    while [ $# -gt 0 ]; do
        case "$1" in
            --host|--port|--model|--quantize|--cache_limit|--device|--device_map|--dtype|--low_vram|--bnb4|--workers)
                args="$args $1 $2"
                shift 2
                ;;
            *)
                args="$args $1"
                shift
                ;;
        esac
    done

    # Start server in background with nohup
    echo "Starting CUDA server with args: $args"
    nohup uv run python server_cuda.py --host $DEFAULT_HOST $args > "$LOG_FILE" 2>&1 &
    local pid=$!
    echo $pid > "$PID_FILE"

    # Wait for server to start
    echo "Waiting for server to start..."
    local max_wait=60  # CUDA may take longer to start
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if is_running && curl -s http://127.0.0.1:$DEFAULT_PORT/api/ps > /dev/null 2>&1; then
            echo "Server started successfully (PID: $pid)"
            echo "API available at: http://127.0.0.1:$DEFAULT_PORT"
            echo "Swagger docs at: http://127.0.0.1:$DEFAULT_PORT/swagger"
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
        echo -n "."
    done
    echo ""
    echo "Server started (PID: $pid) but not yet responding - check logs: $LOG_FILE"
    return 0
}

# Stop the server
stop_server() {
    if ! is_running; then
        echo "Server is not running"
        [ -f "$PID_FILE" ] && rm -f "$PID_FILE"
        return 0
    fi

    local pid=$(get_pid)
    echo "Stopping server (PID: $pid)..."

    # Try graceful shutdown first
    kill $pid 2>/dev/null

    # Wait for process to terminate
    local max_wait=10
    local waited=0
    while [ $waited -lt $max_wait ]; do
        if ! ps -p $pid > /dev/null 2>&1; then
            echo "Server stopped"
            rm -f "$PID_FILE"
            return 0
        fi
        sleep 1
        waited=$((waited + 1))
    done

    # Force kill if still running
    echo "Force killing server..."
    kill -9 $pid 2>/dev/null
    rm -f "$PID_FILE"
    echo "Server stopped"
    return 0
}

# Restart the server
restart_server() {
    stop_server
    sleep 2
    start_server "$@"
}

# Show server status
status_server() {
    if is_running; then
        local pid=$(get_pid)
        echo "Server is running (PID: $pid)"
        echo ""

        # Try to get server info
        if command -v curl > /dev/null 2>&1; then
            echo "Server info:"
            curl -s http://127.0.0.1:$DEFAULT_PORT/api/ps 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "  Unable to fetch server info"
        fi
        return 0
    else
        echo "Server is not running"
        [ -f "$PID_FILE" ] && rm -f "$PID_FILE"
        return 1
    fi
}

# Show usage
show_usage() {
    cat << EOF
mflux-server CUDA control script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
  start       Start the CUDA server (default)
  stop        Stop the server
  restart     Restart the server
  status      Show server status

Options (passed to server):
  --host HOST        Host to bind to (default: $DEFAULT_HOST)
  --port PORT        Port to listen on (default: $DEFAULT_PORT)
  --model MODEL      Model to use (default: $DEFAULT_MODEL)
  --device           Device to use (cuda, cuda:0, cpu)
  --device_map       Device map for multi-GPU
  --dtype            Data type (fp16, bf16, fp32)
  --low_vram         Enable CPU offload
  --bnb4             Enable 4-bit quantization
  --workers N        Number of parallel workers

Examples:
  $0 start                    # Start with defaults
  $0 start --model dev        # Start with dev model
  $0 start --device cuda:0    # Start on specific GPU
  $0 stop                     # Stop the server
  $0 restart --bnb4           # Restart with 4-bit quantization
  $0 status                   # Check status

Logs are written to: $LOG_FILE
EOF
}

# Main script logic
COMMAND="${1:-start}"
shift || true

case "$COMMAND" in
    start)
        start_server "$@"
        ;;
    stop)
        stop_server
        ;;
    restart)
        restart_server "$@"
        ;;
    status)
        status_server
        ;;
    help|--help|-h)
        show_usage
        ;;
    *)
        echo "Unknown command: $COMMAND"
        echo ""
        show_usage
        exit 1
        ;;
esac
