#!/bin/bash

# Zombie Process Killer for Distributed Training Miner

# This script kills orphaned miner.py processes that are not part of the main PM2-managed miner

# Runs in a loop checking every 30 seconds

set -uo pipefail  # Remove -e to allow functions to return error codes without exiting

# Check if running in loop mode (daemon) or single run mode
LOOP_MODE=${LOOP_MODE:-"true"}
CHECK_INTERVAL=${CHECK_INTERVAL:-30}  # seconds

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# PM2 process name
PM2_PROCESS_NAME="distributed_training_miner"
SCRIPT_NAME="neurons/miner.py"

# Function to check if PM2 process is running
check_pm2_process() {
    if ! pm2 list | grep -q "$PM2_PROCESS_NAME.*online"; then
        echo -e "${RED}ERROR: PM2 process '$PM2_PROCESS_NAME' is not running!${NC}"
        return 1
    fi
    echo -e "${GREEN}✓ PM2 process '$PM2_PROCESS_NAME' is running${NC}"
    return 0
}

# Function to get PM2 managed process PID
get_pm2_pid() {
    local pm2_pid_file="/root/.pm2/pids/${PM2_PROCESS_NAME//-/_}-4.pid"
    if [ -f "$pm2_pid_file" ]; then
        cat "$pm2_pid_file"
    else
        # Alternative: get PID from pm2 describe
        pm2 jlist | python3 -c "
import sys, json
data = json.load(sys.stdin)
for p in data:
    if p['name'] == '$PM2_PROCESS_NAME':
        print(p['pid'])
        sys.exit(0)
" 2>/dev/null || echo ""
    fi
}

# Function to get all child PIDs of a parent PID (recursively)
get_child_pids() {
    local parent_pid=$1
    local child_pids=()
    local visited=()
    
    # Recursive function to get all descendants
    get_descendants() {
        local pid=$1
        if [[ " ${visited[@]} " =~ " ${pid} " ]]; then
            return
        fi
        visited+=("$pid")
        
        while IFS= read -r child_pid; do
            if [ -n "$child_pid" ] && [ "$child_pid" != "$parent_pid" ]; then
                child_pids+=("$child_pid")
                get_descendants "$child_pid"
            fi
        done < <(pgrep -P "$pid" 2>/dev/null || true)
    }
    
    get_descendants "$parent_pid"
    echo "${child_pids[@]}"
}

# Function to check if a PID is a direct or indirect child of parent
is_child_of() {
    local pid=$1
    local parent=$2
    local current_ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
    
    while [ -n "$current_ppid" ] && [ "$current_ppid" != "1" ] && [ "$current_ppid" != "0" ]; do
        if [ "$current_ppid" = "$parent" ]; then
            return 0
        fi
        current_ppid=$(ps -o ppid= -p "$current_ppid" 2>/dev/null | tr -d ' ')
    done
    return 1
}

# Function to check if miner is healthy
check_miner_health() {
    local pm2_pid=$1
    local health_ok=true
    
    # Check if PM2 process is still online
    if ! pm2 list | grep -q "$PM2_PROCESS_NAME.*online"; then
        echo -e "${RED}✗ PM2 process went offline!${NC}"
        health_ok=false
    fi
    
    # Check if main process is still running
    if ! ps -p "$pm2_pid" > /dev/null 2>&1; then
        echo -e "${RED}✗ Main process (PID $pm2_pid) is not running!${NC}"
        health_ok=false
    fi
    
    # Check if we still have worker processes
    local worker_count=$(pgrep -P "$pm2_pid" | wc -l)
    if [ "$worker_count" -lt 2 ]; then
        echo -e "${YELLOW}⚠ Warning: Only $worker_count child processes found (expected at least 2)${NC}"
    fi
    
    if [ "$health_ok" = true ]; then
        echo -e "${GREEN}✓ Miner health check passed${NC}"
        return 0
    else
        echo -e "${RED}✗ Miner health check failed!${NC}"
        return 1
    fi
}

# Main execution
main() {
    # Check if PM2 process exists
    if ! check_pm2_process; then
        return 1
    fi
    
    # Get PM2 managed process PID
    local pm2_pid=$(get_pm2_pid)
    if [ -z "$pm2_pid" ]; then
        echo -e "${RED}ERROR: Could not find PM2 managed process PID${NC}"
        return 1
    fi
    
    echo -e "${GREEN}✓ Found PM2 managed process: PID $pm2_pid${NC}"
    
    # Get all legitimate child processes
    local legitimate_pids=($(get_child_pids "$pm2_pid"))
    legitimate_pids+=("$pm2_pid")  # Include main process itself
    
    echo ""
    echo "Legitimate processes (will NOT be killed):"
    for pid in "${legitimate_pids[@]}"; do
        if [ -n "$pid" ]; then
            local cmd=$(ps -p "$pid" -o cmd= 2>/dev/null | head -c 80 || echo "N/A")
            echo "  PID $pid: $cmd"
        fi
    done
    
    # Find all miner.py processes and defunct processes
    echo ""
    echo "Scanning for miner.py processes and defunct processes..."
    local all_miner_pids=($(pgrep -f "$SCRIPT_NAME" 2>/dev/null || true))
    
    # Also find defunct processes that might be related (they won't show up in pgrep)
    # Defunct processes have status 'Z' and are children of our main process
    local defunct_pids=($(ps aux | awk '$8 ~ /^Z/ && $11 ~ /python/ {print $2}' 2>/dev/null || true))
    
    # Filter to only include defunct processes that are children of the PM2 process
    local filtered_defunct=()
    for defunct_pid in "${defunct_pids[@]}"; do
        if is_child_of "$defunct_pid" "$pm2_pid" 2>/dev/null; then
            filtered_defunct+=("$defunct_pid")
        fi
    done
    defunct_pids=("${filtered_defunct[@]}")
    
    # Combine both lists
    all_miner_pids+=("${defunct_pids[@]}")
    
    # Remove duplicates
    local unique_pids=($(printf '%s\n' "${all_miner_pids[@]}" | sort -u))
    all_miner_pids=("${unique_pids[@]}")
    
    if [ ${#all_miner_pids[@]} -eq 0 ]; then
        echo -e "${GREEN}✓ No miner.py processes found${NC}"
        return 0
    fi
    
    # Get the 4 main worker PIDs (should be direct children of torchrun)
    local main_workers=($(pgrep -P "$pm2_pid" -f "$SCRIPT_NAME" 2>/dev/null || true))
    
    # Identify zombie/orphaned processes
    # These are miner.py processes that:
    # 1. Are NOT the main torchrun process
    # 2. Are NOT one of the 4 main worker processes
    # 3. Are defunct (zombie) processes - ALWAYS kill these
    # 4. Are miner.py processes using significant memory
    local zombie_pids=()
    for pid in "${all_miner_pids[@]}"; do
        # Skip if it's the main PM2 process itself
        if [ "$pid" = "$pm2_pid" ]; then
            continue
        fi
        
        # Check process status and command first
        local proc_info=$(ps -p "$pid" -o stat=,cmd= 2>/dev/null || echo "")
        if [ -z "$proc_info" ]; then
            continue  # Process doesn't exist anymore
        fi
        
        local stat=$(echo "$proc_info" | awk '{print $1}')
        local cmd=$(echo "$proc_info" | awk '{$1=""; print substr($0,2)}')
        
        # Check if it's defunct (zombie process) - ALWAYS kill these, regardless of parent
        if echo "$stat" | grep -q "Z"; then
            zombie_pids+=("$pid")
            continue
        fi
        
        # For non-defunct processes, check if they're legitimate workers
        local is_main_worker=false
        for worker_pid in "${main_workers[@]}"; do
            if [ "$pid" = "$worker_pid" ]; then
                is_main_worker=true
                break
            fi
        done
        if [ "$is_main_worker" = true ]; then
            continue  # Don't kill legitimate workers
        fi
        
        # Check if it's a miner.py process using significant memory
        if echo "$cmd" | grep -q "$SCRIPT_NAME"; then
            local mem_kb=$(ps -p "$pid" -o rss= 2>/dev/null | tr -d ' ' || echo "0")
            if [ "$mem_kb" -gt 102400 ]; then  # > 100MB
                zombie_pids+=("$pid")
            fi
        fi
    done
    
    if [ ${#zombie_pids[@]} -eq 0 ]; then
        echo -e "${GREEN}✓ No zombie processes found${NC}"
        return 0
    fi
    
    echo ""
    echo -e "${YELLOW}Found ${#zombie_pids[@]} zombie process(es):${NC}"
    for pid in "${zombie_pids[@]}"; do
        local cmd=$(ps -p "$pid" -o cmd= 2>/dev/null | head -c 100 || echo "N/A")
        local mem=$(ps -p "$pid" -o rss= 2>/dev/null | awk '{printf "%.1f MB", $1/1024}' || echo "N/A")
        echo "  PID $pid: $mem - $cmd"
    done
    
    # Check miner health before killing
    echo ""
    echo "Checking miner health before killing zombies..."
    if ! check_miner_health "$pm2_pid"; then
        echo -e "${RED}ERROR: Miner is not healthy. Aborting zombie kill.${NC}"
        return 1
    fi
    
    # Kill zombie processes
    echo ""
    echo -e "${YELLOW}Killing zombie processes...${NC}"
    local killed_count=0
    local defunct_count=0
    for pid in "${zombie_pids[@]}"; do
        if ! ps -p "$pid" > /dev/null 2>&1; then
            continue  # Process already gone
        fi
        
        # Check if it's defunct (zombie)
        local stat=$(ps -p "$pid" -o stat= 2>/dev/null || echo "")
        if echo "$stat" | grep -q "Z"; then
            echo "  Attempting to clean up defunct process PID $pid..."
            # Defunct processes can't be killed directly - they need parent to reap them
            # Try sending SIGCHLD to parent to trigger reaping (if parent is our process)
            local ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
            if [ -n "$ppid" ] && is_child_of "$ppid" "$pm2_pid" 2>/dev/null; then
                # Parent is part of our process tree - try to trigger reaping
                kill -CHLD "$ppid" 2>/dev/null || true
            fi
            defunct_count=$((defunct_count + 1))
        else
            echo "  Killing PID $pid..."
            kill -9 "$pid" 2>/dev/null && killed_count=$((killed_count + 1)) || echo "    Failed to kill PID $pid"
        fi
        sleep 0.5
    done
    
    if [ $killed_count -gt 0 ]; then
        echo -e "${GREEN}✓ Killed $killed_count zombie process(es)${NC}"
    fi
    if [ $defunct_count -gt 0 ]; then
        echo -e "${YELLOW}⚠ Note: $defunct_count defunct process(es) found. These are already dead and use 0 MB memory."
        echo "  They will be cleaned up when the parent process reaps them.${NC}"
    fi
    
    # Wait a moment for processes to terminate
    sleep 2
    
    # Check miner health after killing
    echo ""
    echo "Checking miner health after killing zombies..."
    if check_miner_health "$pm2_pid"; then
        echo ""
        echo -e "${GREEN}=== SUCCESS ===${NC}"
        echo "Zombie processes killed successfully. Main miner is healthy."
        return 0
    else
        echo ""
        echo -e "${RED}=== WARNING ===${NC}"
        echo "Miner health check failed after killing zombies!"
        echo "The main miner may have been affected. Please check manually."
        return 1
    fi
}

# Run main function
if [ "$LOOP_MODE" = "true" ]; then
    # Run in loop mode (daemon)
    echo -e "${GREEN}=== Zombie Process Killer (Loop Mode) ===${NC}"
    echo -e "${GREEN}Checking every ${CHECK_INTERVAL} seconds${NC}"
    echo ""
    
    while true; do
        main "$@"
        local exit_code=$?
        
        # If main returns non-zero, wait a bit longer before retrying
        if [ $exit_code -ne 0 ]; then
            sleep 5
        else
            sleep "$CHECK_INTERVAL"
        fi
    done
else
    # Run once (for cron jobs)
    echo -e "${GREEN}=== Zombie Process Killer ===${NC}"
    echo ""
    main "$@"
fi

