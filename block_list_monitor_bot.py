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
    
    def check_nvidia_status(self):
        """Check NVIDIA GPU status and return GPU information"""
        gpu_info = []
        try:
            self.log("📊 Checking NVIDIA GPU status...")
            nvidia_result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if nvidia_result.returncode == 0:
                lines = [l.strip() for l in nvidia_result.stdout.strip().split('\n') if l.strip()]
                self.log(f"   📊 Found {len(lines)} GPU(s)")
                for line in lines:
                    parts = line.split(',')
                    if len(parts) >= 3:
                        try:
                            gpu_id = parts[0].strip()
                            used = int(parts[1].strip())
                            total = int(parts[2].strip())
                            gpu_info.append({"id": gpu_id, "used": used, "total": total})
                            self.log(f"      GPU {gpu_id}: {used}MB / {total}MB used")
                        except:
                            pass
            else:
                self.log("   ⚠️  nvidia-smi returned non-zero exit code")
        except FileNotFoundError:
            self.log("   ❌ nvidia-smi not found - NVIDIA drivers may not be installed")
        except Exception as e:
            self.log(f"   ⚠️  Could not check GPU status: {e}")
        return gpu_info
    
    def kill_all_zombie_processes(self):
        """Kill all zombie processes related to training/mining"""
        self.log("🧹 Killing all zombie processes...")
        
        zombie_patterns = [
            "python.*miner",
            "python.*training",
            "torchrun",
            "hivemind",
            "distributed_training",
            "python.*neurons",
            "python.*distributed"
        ]
        
        killed_count = 0
        for pattern in zombie_patterns:
            try:
                # First, find processes
                pgrep_result = subprocess.run(
                    ["pgrep", "-f", pattern],
                    capture_output=True,
                    text=True,
                    timeout=3,
                    stderr=subprocess.DEVNULL
                )
                
                if pgrep_result.returncode == 0 and pgrep_result.stdout.strip():
                    pids = [pid.strip() for pid in pgrep_result.stdout.strip().split('\n') if pid.strip()]
                    for pid in pids:
                        try:
                            # Try graceful kill first
                            subprocess.run(
                                ["kill", "-TERM", pid],
                                capture_output=True,
                                timeout=1,
                                stderr=subprocess.DEVNULL
                            )
                            time.sleep(0.5)
                            # Force kill if still running
                            subprocess.run(
                                ["kill", "-9", pid],
                                capture_output=True,
                                timeout=1,
                                stderr=subprocess.DEVNULL
                            )
                            killed_count += 1
                            self.log(f"   ✅ Killed zombie process: PID {pid} (pattern: {pattern})")
                        except:
                            pass
            except FileNotFoundError:
                # pgrep not available, use pkill
                try:
                    pkill_result = subprocess.run(
                        ["pkill", "-9", "-f", pattern],
                        capture_output=True,
                        timeout=3,
                        stderr=subprocess.DEVNULL
                    )
                    if pkill_result.returncode == 0:
                        killed_count += 1
                        self.log(f"   ✅ Killed processes matching pattern: {pattern}")
                except:
                    pass
            except Exception as e:
                self.log(f"   ⚠️  Error killing processes for pattern {pattern}: {e}")
        
        # Also kill any orphaned Python processes that might be holding resources
        try:
            ps_result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if ps_result.returncode == 0:
                for line in ps_result.stdout.split('\n'):
                    if 'python' in line.lower() and ('miner' in line.lower() or 'training' in line.lower() or 'torchrun' in line.lower()):
                        parts = line.split()
                        if len(parts) >= 2:
                            pid = parts[1]
                            if pid.isdigit():
                                try:
                                    subprocess.run(["kill", "-9", pid], capture_output=True, timeout=1, stderr=subprocess.DEVNULL)
                                    killed_count += 1
                                    self.log(f"   ✅ Killed orphaned process: PID {pid}")
                                except:
                                    pass
        except:
            pass
        
        if killed_count == 0:
            self.log("   ℹ️  No zombie processes found")
        else:
            self.log(f"   ✅ Killed {killed_count} zombie process(es)")
        
        return killed_count
    
    def clear_gpu_processes(self):
        """Clear ALL GPU processes on ALL GPUs before restart to prevent CUDA OOM errors"""
        try:
            self.log("🧹 Clearing ALL GPU processes before restart...")
            
            # Step 1: Get all processes using GPU via nvidia-smi
            gpu_pids = set()
            try:
                nvidia_result = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if nvidia_result.returncode == 0 and nvidia_result.stdout.strip():
                    lines = [l.strip() for l in nvidia_result.stdout.strip().split('\n') if l.strip()]
                    for line in lines:
                        parts = line.split(',')
                        if len(parts) >= 1:
                            pid = parts[0].strip()
                            if pid.isdigit():
                                gpu_pids.add(pid)
                                process_name = parts[1].strip() if len(parts) > 1 else "[Unknown]"
                                memory = parts[2].strip() if len(parts) > 2 else "[Unknown]"
                                self.log(f"   🔍 Found process using GPU: PID {pid} ({process_name}) - {memory}")
            except FileNotFoundError:
                self.log("   ⚠️  nvidia-smi not found, will use alternative methods")
            except Exception as e:
                self.log(f"   ⚠️  Error checking nvidia-smi: {e}")
            
            
            # Step 2: Kill all processes using GPU devices directly (including nvidia-uvm, nvidiactl)
            try:
                # Check all GPU device files
                gpu_devices = []
                # Check numbered GPU devices (nvidia0, nvidia1, etc.)
                for device_num in range(8):  # Check up to 8 GPUs
                    device = f"/dev/nvidia{device_num}"
                    if os.path.exists(device):
                        gpu_devices.append(device)
                
                # Also check common NVIDIA device files
                for device_name in ["/dev/nvidia-uvm", "/dev/nvidiactl"]:
                    if os.path.exists(device_name):
                        gpu_devices.append(device_name)
                
                # Kill processes using each device
                for device in gpu_devices:
                    try:
                        # First, find processes using this device
                        fuser_result = subprocess.run(
                            ["fuser", device],
                            capture_output=True,
                            text=True,
                            stderr=subprocess.DEVNULL,
                            timeout=2
                        )
                        if fuser_result.returncode == 0:
                            # Extract PIDs from fuser output
                            pids = [p.strip() for p in fuser_result.stdout.strip().split() if p.strip().isdigit()]
                            gpu_pids.update(pids)
                            
                            # Kill processes using this device
                            subprocess.run(
                                ["fuser", "-k", device],
                                capture_output=True,
                                stderr=subprocess.DEVNULL,
                                timeout=2
                            )
                            if pids:
                                self.log(f"   ✅ Killed processes using {device}: {', '.join(pids)}")
                    except:
                        pass
            except Exception as e:
                self.log(f"   ⚠️  Error checking GPU devices: {e}")
            
            # Step 3: Kill all GPU processes found by PID
            if gpu_pids:
                for pid in gpu_pids:
                    try:
                        subprocess.run(
                            ["kill", "-9", pid],
                            capture_output=True,
                            timeout=2,
                            stderr=subprocess.DEVNULL
                        )
                        self.log(f"   ✅ Killed GPU process {pid}")
                    except Exception as e:
                        self.log(f"   ⚠️  Could not kill process {pid}: {e}")
            
            # Step 4: Wait for processes to release GPU memory
            self.log("   ⏳ Waiting for GPU memory to be released...")
            time.sleep(3)
            
            # Step 5: Verify ALL GPUs are clear (with retries)
            max_retries = 5
            all_gpus_clear = False
            for attempt in range(max_retries):
                try:
                    nvidia_result = subprocess.run(
                        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    if nvidia_result.returncode == 0:
                        lines = [l.strip() for l in nvidia_result.stdout.strip().split('\n') if l.strip()]
                        all_clear = True
                        max_used = 0
                        
                        for line in lines:
                            parts = line.split(',')
                            if len(parts) >= 3:
                                try:
                                    gpu_id = parts[0].strip()
                                    used = int(parts[1].strip())
                                    total = int(parts[2].strip())
                                    max_used = max(max_used, used)
                                    
                                    # Consider GPU clear if less than 10MB used (driver overhead)
                                    if used >= 10:
                                        all_clear = False
                                except:
                                    pass
                        
                        if all_clear:
                            self.log(f"   ✅ ALL GPUs cleared successfully (max memory used: {max_used}MB)")
                            all_gpus_clear = True
                            break
                        else:
                            if attempt < max_retries - 1:
                                self.log(f"   ⚠️  Some GPUs still have memory allocated (max: {max_used}MB), retrying... (attempt {attempt + 1}/{max_retries})")
                                # Try killing any remaining processes
                                try:
                                    nvidia_result = subprocess.run(
                                        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                                        capture_output=True,
                                        text=True,
                                        timeout=5
                                    )
                                    if nvidia_result.returncode == 0 and nvidia_result.stdout.strip():
                                        pids = [pid.strip() for pid in nvidia_result.stdout.strip().split('\n') if pid.strip()]
                                        for pid in pids:
                                            try:
                                                subprocess.run(["kill", "-9", pid], capture_output=True, timeout=1, stderr=subprocess.DEVNULL)
                                            except:
                                                pass
                                except:
                                    pass
                                time.sleep(2)
                            else:
                                self.log(f"   ⚠️  Some GPUs still have memory allocated (max: {max_used}MB) after {max_retries} attempts")
                except Exception as e:
                    if attempt < max_retries - 1:
                        self.log(f"   ⚠️  Could not verify GPU status, retrying... (attempt {attempt + 1}/{max_retries})")
                        time.sleep(2)
                    else:
                        self.log(f"   ⚠️  Could not verify GPU status: {e}")
            
            # Step 6: Final verification - check for any remaining processes
            try:
                nvidia_result = subprocess.run(
                    ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if nvidia_result.returncode == 0 and nvidia_result.stdout.strip():
                    remaining_pids = [pid.strip() for pid in nvidia_result.stdout.strip().split('\n') if pid.strip()]
                    if remaining_pids:
                        self.log(f"   ⚠️  Warning: {len(remaining_pids)} process(es) still using GPU: {', '.join(remaining_pids)}")
                    else:
                        self.log("   ✅ Verified: No processes using GPU")
            except:
                pass
                
        except Exception as e:
            self.log(f"   ⚠️  Error during GPU cleanup: {e}")
    
    def restart_pm2_process(self):
        """Restart the pm2 process with proper cleanup: check nvidia -> clear GPU -> kill zombies -> verify -> restart"""
        current_time = time.time()
        
        # Check cooldown to prevent rapid restarts
        if current_time - self.last_restart_time < RESTART_COOLDOWN:
            remaining = int(RESTART_COOLDOWN - (current_time - self.last_restart_time))
            self.log(f"⏳ Restart cooldown active. {remaining}s remaining. Skipping restart.")
            return False
        
        try:
            self.log("=" * 70)
            self.log(f"⚠️  Trigger detected! Preparing to restart {self.process_name}...")
            self.log("=" * 70)
            
            # Step 1: Check NVIDIA status first
            gpu_info = self.check_nvidia_status()
            
            # Step 2: Clear all GPU processes
            self.clear_gpu_processes()
            
            # Step 3: Kill all zombie processes
            self.kill_all_zombie_processes()
            
            # Step 4: Wait a bit for everything to settle
            self.log("   ⏳ Waiting for processes to fully terminate...")
            time.sleep(2)
            
            # Step 5: Final verification - check NVIDIA status again
            self.log("   🔍 Verifying cleanup...")
            final_gpu_info = self.check_nvidia_status()
            
            # Check if GPUs are clear (less than 10MB used per GPU)
            all_clear = True
            for gpu in final_gpu_info:
                if gpu.get("used", 0) >= 10:
                    all_clear = False
                    break
            
            if all_clear:
                self.log("   ✅ All GPUs cleared successfully")
            else:
                self.log("   ⚠️  Some GPUs may still have memory allocated, but proceeding with restart...")
            
            # Step 6: Restart the PM2 process
            self.log("=" * 70)
            self.log(f"🔄 Restarting {self.process_name}...")
            self.log("=" * 70)
            
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
            
            self.log("=" * 70)
            self.log(f"✅ Successfully restarted {self.process_name} (restart #{self.restart_count})")
            self.log(f"⏳ Waiting {RESTART_COOLDOWN} seconds before resuming monitoring...")
            self.log("=" * 70)
            time.sleep(RESTART_COOLDOWN)
            
            # Reset log position after restart - will reinitialize to end of file
            self.initialize_log_position()
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.log("=" * 70)
            self.log(f"❌ Error restarting process: {e}")
            self.log(f"Error output: {e.stderr if e.stderr else 'None'}")
            self.log("=" * 70)
            return False
        except subprocess.TimeoutExpired:
            self.log("=" * 70)
            self.log("❌ Timeout while restarting process")
            self.log("=" * 70)
            return False
        except Exception as e:
            self.log("=" * 70)
            self.log(f"❌ Unexpected error during restart: {e}")
            self.log("=" * 70)
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

