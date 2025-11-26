#!/usr/bin/env python3
"""
PM2 Auto-Restart Bot
Monitors pm2 logs for distributed_training_miner and restarts when specific log messages are detected.
"""

import subprocess
import time
import re
from datetime import datetime
import sys

# Configuration
PM2_PROCESS_NAME = "distributed_training_miner"
TRIGGER_MESSAGE = "New epoch model may not be available yet. Using robust retry mechanism"
CHECK_INTERVAL = 2  # seconds between checks
RESTART_COOLDOWN = 30  # seconds to wait after restart before monitoring again
RESTART_DELAY = 300  # seconds (5 minutes) to wait after trigger detected before restarting

class PM2MonitorBot:
    def __init__(self, process_name, trigger_message):
        self.process_name = process_name
        self.trigger_message = trigger_message
        self.last_restart_time = 0
        self.restart_count = 0
        
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
    
    def restart_pm2_process(self):
        """Restart the pm2 process"""
        current_time = time.time()
        
        # Check cooldown to prevent rapid restarts
        if current_time - self.last_restart_time < RESTART_COOLDOWN:
            self.log(f"Restart cooldown active. Skipping restart.")
            return False
        
        try:
            self.log(f"⚠️  Trigger detected! Restarting {self.process_name}...")
            
            # Restart the process
            result = subprocess.run(
                ["pm2", "restart", self.process_name],
                capture_output=True,
                text=True,
                check=True
            )
            
            self.last_restart_time = current_time
            self.restart_count += 1
            
            self.log(f"✓ Successfully restarted {self.process_name} (restart #{self.restart_count})")
            self.log(f"Waiting {RESTART_COOLDOWN} seconds before resuming monitoring...")
            time.sleep(RESTART_COOLDOWN)
            
            return True
            
        except subprocess.CalledProcessError as e:
            self.log(f"❌ Error restarting process: {e}")
            self.log(f"Error output: {e.stderr if e.stderr else 'None'}")
            return False
    
    def get_recent_logs(self, lines=50):
        """Get recent logs from pm2"""
        try:
            result = subprocess.run(
                ["pm2", "logs", self.process_name, "--nostream", "--lines", str(lines)],
                capture_output=True,
                text=True,
                check=True,
                timeout=10
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            self.log(f"Error getting logs: {e}")
            return ""
        except subprocess.TimeoutExpired:
            self.log("Timeout getting logs")
            return ""
    
    def monitor_logs_stream(self):
        """Monitor pm2 logs in real-time using streaming"""
        try:
            self.log(f"Starting real-time log monitoring for {self.process_name}...")
            
            # Start pm2 logs in streaming mode
            process = subprocess.Popen(
                ["pm2", "logs", self.process_name, "--nostream", "--lines", "0"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Now stream new logs
            stream_process = subprocess.Popen(
                ["pm2", "logs", self.process_name, "--lines", "0"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            self.log(f"Monitoring active. Watching for: '{self.trigger_message}'")
            
            for line in stream_process.stdout:
                line = line.strip()
                if line:
                    # Check if trigger message is in the log line
                    if self.trigger_message in line:
                        self.log(f"🔍 Detected trigger message in log:")
                        self.log(f"   {line}")
                        self.log(f"⏳ Waiting {RESTART_DELAY} seconds (5 minutes) before restarting...")
                        
                        # Terminate the streaming process before waiting
                        stream_process.terminate()
                        stream_process.wait(timeout=5)
                        
                        # Wait for the specified delay
                        time.sleep(RESTART_DELAY)
                        
                        # Restart the process
                        if self.restart_pm2_process():
                            # Start monitoring again
                            return self.monitor_logs_stream()
                        else:
                            # If restart failed, continue monitoring
                            return self.monitor_logs_stream()
            
            # If stream ended, restart monitoring
            return self.monitor_logs_stream()
            
        except KeyboardInterrupt:
            self.log("Monitoring stopped by user")
            raise
        except Exception as e:
            self.log(f"Error in log monitoring: {e}")
            time.sleep(5)
            return self.monitor_logs_stream()
    
    def run(self):
        """Main monitoring loop"""
        self.log("=" * 70)
        self.log(f"PM2 Auto-Restart Bot Started")
        self.log(f"Process: {self.process_name}")
        self.log(f"Trigger: {self.trigger_message}")
        self.log(f"Restart Delay: {RESTART_DELAY} seconds (5 minutes)")
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
        
        try:
            # Start monitoring
            self.monitor_logs_stream()
            
        except KeyboardInterrupt:
            self.log("\n" + "=" * 70)
            self.log(f"Bot stopped. Total restarts performed: {self.restart_count}")
            self.log("=" * 70)

def main():
    bot = PM2MonitorBot(PM2_PROCESS_NAME, TRIGGER_MESSAGE)
    bot.run()

if __name__ == "__main__":
    main()

