#!/usr/bin/env python3
"""
Block List Monitor Bot
Monitors pm2 logs for distributed_training_miner and restarts when "block_list: []" is detected.
"""

import subprocess
import time
import re
from datetime import datetime
import sys
import os

# Configuration
PM2_PROCESS_NAME = "distributed_training_miner"
TRIGGER_PATTERN = r"block_list:\s*\[\]"  # Matches "block_list: []"
CHECK_INTERVAL = 1  # seconds between log checks
RESTART_COOLDOWN = 60  # seconds to wait after restart before monitoring again
LOG_FILE_PATH = "/root/.pm2/logs/distributed-training-miner-out.log"

class BlockListMonitorBot:
    def __init__(self, process_name, trigger_pattern):
        self.process_name = process_name
        self.trigger_pattern = trigger_pattern
        self.last_restart_time = 0
        self.restart_count = 0
        self.last_checked_position = 0
        self.trigger_detected = False
        
    def log(self, message):
        """Print timestamped log message"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {message}")
        sys.stdout.flush()
    
    def check_pm2_process_exists(self):
        """Check if the pm2 process exists"""
        try:
            result = subprocess.run(
                ["pm2", "jlist"],
                capture_output=True,
                text=True,
                check=True
            )
            return self.process_name in result.stdout
        except subprocess.CalledProcessError as e:
            self.log(f"Error checking pm2 processes: {e}")
            return False
        except FileNotFoundError:
            self.log("❌ Error: pm2 command not found")
            return False
    
    def clear_gpu_processes(self):
        """Clear all GPU processes (zombie processes) before restart"""
        try:
            self.log("🧹 Clearing GPU processes...")
            
            # Check GPU usage
            try:
                nvidia_result = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if nvidia_result.returncode == 0 and nvidia_result.stdout.strip():
                    pids = [pid.strip() for pid in nvidia_result.stdout.strip().split('\n') if pid.strip()]
                    if pids:
                        self.log(f"🔍 Found {len(pids)} process(es) using GPU: {', '.join(pids)}")
                        
                        # Kill processes using GPU
                        for pid in pids:
                            try:
                                subprocess.run(
                                    ["kill", "-9", pid],
                                    capture_output=True,
                                    timeout=2
                                )
                                self.log(f"   ✅ Killed process {pid}")
                            except Exception as e:
                                self.log(f"   ⚠️  Could not kill process {pid}: {e}")
                else:
                    self.log("   ℹ️  No processes found using GPU")
            except FileNotFoundError:
                self.log("   ⚠️  nvidia-smi not found, skipping GPU process check")
            except subprocess.TimeoutExpired:
                self.log("   ⚠️  Timeout checking GPU processes")
            except Exception as e:
                self.log(f"   ⚠️  Error checking GPU: {e}")
            
            # Kill any Python/torchrun processes that might be holding GPU
            try:
                # Kill Python processes related to miner/training
                pkill_result = subprocess.run(
                    ["pkill", "-9", "-f", "python.*miner|torchrun|hivemind"],
                    capture_output=True,
                    timeout=3
                )
                
                # Also try to kill processes using GPU devices directly
                try:
                    # Try to find and kill processes using /dev/nvidia devices
                    for device in ["/dev/nvidia0", "/dev/nvidia1", "/dev/nvidia2", "/dev/nvidia3"]:
                        try:
                            fuser_result = subprocess.run(
                                ["fuser", "-k", device],
                                capture_output=True,
                                stderr=subprocess.DEVNULL,
                                timeout=2
                            )
                        except:
                            pass
                except:
                    pass
                
                if pkill_result.returncode == 0 or pkill_result.returncode == 1:  # 1 means no processes found
                    if pkill_result.returncode == 0:
                        self.log("   ✅ Killed zombie Python/torchrun processes")
                    else:
                        self.log("   ℹ️  No zombie processes found to kill")
            except Exception as e:
                self.log(f"   ⚠️  Error killing zombie processes: {e}")
            
            # Wait a moment for processes to release GPU
            time.sleep(2)
            
            # Verify GPU is clear
            try:
                nvidia_result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if nvidia_result.returncode == 0:
                    memory_usage = nvidia_result.stdout.strip().split('\n')
                    total_usage = sum(int(usage.strip()) for usage in memory_usage if usage.strip().isdigit())
                    if total_usage < 1000:  # Less than 1GB total across all GPUs
                        self.log(f"   ✅ GPU cleared successfully (total memory: {total_usage}MB)")
                    else:
                        self.log(f"   ⚠️  GPU still has {total_usage}MB allocated")
            except Exception as e:
                self.log(f"   ⚠️  Could not verify GPU status: {e}")
                
        except Exception as e:
            self.log(f"   ⚠️  Error during GPU cleanup: {e}")
    
    def restart_pm2_process(self):
        """Restart the pm2 process"""
        current_time = time.time()
        
        # Check cooldown to prevent rapid restarts
        if current_time - self.last_restart_time < RESTART_COOLDOWN:
            remaining = int(RESTART_COOLDOWN - (current_time - self.last_restart_time))
            self.log(f"⏳ Restart cooldown active. {remaining}s remaining. Skipping restart.")
            return False
        
        try:
            self.log(f"⚠️  Trigger detected! Restarting {self.process_name}...")
            
            # Clear GPU processes before restart
            self.clear_gpu_processes()
            
            # Restart the process
            result = subprocess.run(
                ["pm2", "restart", self.process_name],
                capture_output=True,
                text=True,
                check=True,
                timeout=30
            )
            
            self.last_restart_time = current_time
            self.restart_count += 1
            self.trigger_detected = False  # Reset trigger flag after restart
            
            self.log(f"✅ Successfully restarted {self.process_name} (restart #{self.restart_count})")
            self.log(f"⏳ Waiting {RESTART_COOLDOWN} seconds before resuming monitoring...")
            time.sleep(RESTART_COOLDOWN)
            
            # Reset log position after restart - will reinitialize to end of file
            self.initialize_log_position()
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.log(f"❌ Error restarting process: {e}")
            self.log(f"Error output: {e.stderr if e.stderr else 'None'}")
            return False
        except subprocess.TimeoutExpired:
            self.log("❌ Timeout while restarting process")
            return False
    
    def initialize_log_position(self):
        """Initialize log position to current end of file (only monitor NEW logs)"""
        try:
            if os.path.exists(LOG_FILE_PATH):
                # Set position to end of file - only monitor NEW logs from now on
                with open(LOG_FILE_PATH, 'r', encoding='utf-8', errors='ignore') as f:
                    f.seek(0, 2)  # Seek to end of file
                    self.last_checked_position = f.tell()
                    self.log(f"✅ Initialized log position to end of file (position: {self.last_checked_position})")
                    self.log("📋 Only monitoring NEW logs from this point forward")
            else:
                self.last_checked_position = 0
                self.log(f"⚠️  Log file not found, will start from beginning when created")
        except Exception as e:
            self.log(f"❌ Error initializing log position: {e}")
            self.last_checked_position = 0
    
    def monitor_log_file(self):
        """Monitor log file for trigger pattern - ONLY NEW logs after bot start"""
        try:
            # Check if log file exists
            if not os.path.exists(LOG_FILE_PATH):
                self.log(f"⚠️  Log file not found: {LOG_FILE_PATH}")
                self.log("Waiting for log file to be created...")
                time.sleep(10)
                return
            
            # Open and read from last checked position
            with open(LOG_FILE_PATH, 'r', encoding='utf-8', errors='ignore') as f:
                # Get current file size
                f.seek(0, 2)  # Seek to end
                current_size = f.tell()
                
                # If file was truncated or rotated (current size < last position), reset position
                if current_size < self.last_checked_position:
                    self.log("⚠️  Log file was rotated/truncated, resetting position")
                    self.last_checked_position = 0
                
                # Seek to last checked position
                try:
                    f.seek(self.last_checked_position)
                except (IOError, OSError):
                    # If seek fails, reset position
                    self.last_checked_position = 0
                    f.seek(0)
                
                # Read ONLY new lines (lines written after last check)
                new_lines = f.readlines()
                
                # Update position to current end of file
                self.last_checked_position = f.tell()
                
                # Check each NEW line for trigger pattern
                for line in new_lines:
                    if re.search(self.trigger_pattern, line, re.IGNORECASE):
                        if not self.trigger_detected:  # Only log once per detection
                            self.log(f"🔍 Detected trigger pattern in NEW log:")
                            self.log(f"   {line.strip()}")
                            self.trigger_detected = True
                            
                            # Restart immediately when detected
                            if self.restart_pm2_process():
                                return
                            else:
                                # If restart failed, continue monitoring
                                self.trigger_detected = False
                                return
                
        except FileNotFoundError:
            self.log(f"⚠️  Log file not found: {LOG_FILE_PATH}")
            time.sleep(5)
        except PermissionError:
            self.log(f"❌ Permission denied reading log file: {LOG_FILE_PATH}")
            time.sleep(10)
        except Exception as e:
            self.log(f"❌ Error monitoring log file: {e}")
            time.sleep(5)
    
    def monitor_logs_pm2(self):
        """Alternative: Monitor using pm2 logs command"""
        try:
            # Get recent logs from pm2
            result = subprocess.run(
                ["pm2", "logs", self.process_name, "--nostream", "--lines", "50"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10
            )
            
            logs = result.stdout + result.stderr
            
            # Check for trigger pattern
            if re.search(self.trigger_pattern, logs, re.IGNORECASE):
                if not self.trigger_detected:
                    self.log(f"🔍 Detected trigger pattern in logs:")
                    # Extract the matching line
                    for line in logs.split('\n'):
                        if re.search(self.trigger_pattern, line, re.IGNORECASE):
                            self.log(f"   {line.strip()}")
                            break
                    self.trigger_detected = True
                    
                    # Restart immediately when detected
                    if self.restart_pm2_process():
                        return
                    else:
                        self.trigger_detected = False
                        return
            else:
                # Reset trigger flag if pattern not found
                self.trigger_detected = False
                
        except subprocess.CalledProcessError as e:
            self.log(f"Error getting logs from pm2: {e}")
        except subprocess.TimeoutExpired:
            self.log("Timeout getting logs from pm2")
        except Exception as e:
            self.log(f"Error in pm2 log monitoring: {e}")
    
    def run(self):
        """Main monitoring loop"""
        self.log("=" * 70)
        self.log(f"Block List Monitor Bot Started")
        self.log(f"Process: {self.process_name}")
        self.log(f"Trigger Pattern: {self.trigger_pattern}")
        self.log(f"Log File: {LOG_FILE_PATH}")
        self.log(f"Check Interval: {CHECK_INTERVAL} seconds")
        self.log(f"Restart Cooldown: {RESTART_COOLDOWN} seconds")
        self.log("=" * 70)
        
        # Check if pm2 is installed
        try:
            subprocess.run(["pm2", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.log("❌ Error: pm2 is not installed or not in PATH")
            return
        
        # Check if process exists
        if not self.check_pm2_process_exists():
            self.log(f"⚠️  Warning: Process '{self.process_name}' not found in pm2 list")
            self.log("Will continue monitoring anyway...")
        
        self.log(f"✅ Monitoring started. Watching for: '{self.trigger_pattern}'")
        self.log("Press Ctrl+C to stop")
        self.log("-" * 70)
        
        # Initialize log position to END of file - only monitor NEW logs from now
        self.initialize_log_position()
        
        try:
            while True:
                # Try monitoring log file first (more efficient)
                if os.path.exists(LOG_FILE_PATH):
                    self.monitor_log_file()
                else:
                    # Fallback to pm2 logs command (but this checks old logs too)
                    # Only use as fallback
                    self.log("⚠️  Using pm2 logs fallback (may check old logs)")
                    self.monitor_logs_pm2()
                
                time.sleep(CHECK_INTERVAL)
                
        except KeyboardInterrupt:
            self.log("\n" + "=" * 70)
            self.log(f"Bot stopped by user")
            self.log(f"Total restarts performed: {self.restart_count}")
            self.log("=" * 70)
        except Exception as e:
            self.log(f"❌ Unexpected error: {e}")
            import traceback
            self.log(traceback.format_exc())
            time.sleep(5)
            # Restart monitoring
            self.run()

def main():
    bot = BlockListMonitorBot(PM2_PROCESS_NAME, TRIGGER_PATTERN)
    bot.run()

if __name__ == "__main__":
    main()

