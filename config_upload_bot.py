#!/usr/bin/env python3
"""
Config Upload Bot - Uploads config.json every 30 seconds
Keeps block_list fresh for validator time checking without uploading full model.
"""

import os
import sys
import time
import json
import logging
import boto3
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Configuration
UPLOAD_INTERVAL = 30  # 30 seconds
CONFIG_FILE = "config.json"
CHECK_INTERVAL = 2  # Check for new config every 2 seconds

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('config_upload_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ConfigUploadBot:
    def __init__(self):
        """Initialize the Config Upload Bot"""
        self.bucket_name = self._find_bucket_name()
        self.output_dir = self._find_output_dir()
        self.config_path = os.path.join(self.output_dir, CONFIG_FILE)
        
        # Initialize R2 client
        self.r2_client = self._init_r2_client()
        
        # Track last uploaded config
        self.last_uploaded_content = None
        self.last_upload_time = 0
        
        logger.info("=" * 70)
        logger.info("🚀 Config Upload Bot Initialized")
        logger.info(f"📦 Bucket: {self.bucket_name}")
        logger.info(f"📁 Output Dir: {self.output_dir}")
        logger.info(f"📄 Config File: {self.config_path}")
        logger.info(f"⏱️  Upload Interval: {UPLOAD_INTERVAL} seconds")
        logger.info("=" * 70)
    
    def _find_bucket_name(self):
        """Find the R2 bucket name from environment"""
        bucket_name = os.getenv('R2_BUCKET_NAME')
        if not bucket_name:
            logger.error("❌ R2_BUCKET_NAME not found in environment variables")
            sys.exit(1)
        return bucket_name
    
    def _find_output_dir(self):
        """Find the output directory containing config.json"""
        # Check common locations
        possible_paths = [
            f'/workspace/s38/{self.bucket_name}',
            f'/workspace/s38/llama-4b-ws-4-227',
            '/workspace/s38/output',
        ]
        
        for path in possible_paths:
            config_file = os.path.join(path, CONFIG_FILE)
            if os.path.exists(config_file):
                logger.info(f"✅ Found config.json at: {path}")
                return path
        
        # If not found, use the bucket name directory
        default_path = f'/workspace/s38/{self.bucket_name}'
        logger.warning(f"⚠️  Config.json not found yet. Will monitor: {default_path}")
        return default_path
    
    def _init_r2_client(self):
        """Initialize R2 client with write credentials"""
        account_id = os.getenv('R2_ACCOUNT_ID')
        access_key = os.getenv('R2_WRITE_ACCESS_KEY_ID')
        secret_key = os.getenv('R2_WRITE_SECRET_ACCESS_KEY')
        
        if not all([account_id, access_key, secret_key]):
            logger.error("❌ Missing R2 write credentials in environment variables")
            sys.exit(1)
        
        return boto3.client(
            's3',
            endpoint_url=f'https://{account_id}.r2.cloudflarestorage.com',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name='auto'
        )
    
    def read_config(self):
        """Read the current config.json file"""
        try:
            if not os.path.exists(self.config_path):
                return None
            
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ Error reading config.json: {e}")
            return None
    
    def has_config_changed(self, current_config):
        """Check if config has changed since last upload"""
        if current_config is None:
            return False
        
        current_content = json.dumps(current_config, sort_keys=True)
        
        if self.last_uploaded_content is None:
            return True
        
        return current_content != self.last_uploaded_content
    
    def should_upload_by_time(self):
        """Check if enough time has passed since last upload"""
        current_time = time.time()
        time_since_upload = current_time - self.last_upload_time
        return time_since_upload >= UPLOAD_INTERVAL
    
    def upload_config(self, config_data):
        """Upload config.json to R2 bucket"""
        try:
            # Convert to JSON string
            config_json = json.dumps(config_data, indent=2)
            
            # Get block_list info for logging
            block_list = config_data.get('block_list', [])
            block_info = f"{len(block_list)} blocks"
            if block_list:
                latest_blocks = block_list[-3:] if len(block_list) > 3 else block_list
                block_info += f", latest: {latest_blocks}"
            
            logger.info(f"📤 Uploading config.json ({block_info})...")
            
            # Upload to R2
            self.r2_client.put_object(
                Bucket=self.bucket_name,
                Key=CONFIG_FILE,
                Body=config_json.encode('utf-8'),
                ContentType='application/json'
            )
            
            # Update tracking
            self.last_uploaded_content = json.dumps(config_data, sort_keys=True)
            self.last_upload_time = time.time()
            
            logger.info(f"✅ Successfully uploaded config.json")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to upload config.json: {e}")
            return False
    
    def run(self):
        """Main bot loop"""
        logger.info("🔄 Starting config upload loop...")
        
        consecutive_failures = 0
        max_failures = 10
        
        while True:
            try:
                # Read current config
                config_data = self.read_config()
                
                if config_data is None:
                    logger.debug(f"⏳ Waiting for config.json to be created at {self.config_path}...")
                    time.sleep(CHECK_INTERVAL)
                    continue
                
                # Check if we should upload
                config_changed = self.has_config_changed(config_data)
                time_to_upload = self.should_upload_by_time()
                
                if config_changed or time_to_upload:
                    reason = "config changed" if config_changed else "time interval reached"
                    logger.info(f"🔔 Upload triggered: {reason}")
                    
                    success = self.upload_config(config_data)
                    
                    if success:
                        consecutive_failures = 0
                        next_upload = time.strftime('%H:%M:%S', time.localtime(time.time() + UPLOAD_INTERVAL))
                        logger.info(f"⏳ Next upload at {next_upload} (or when config changes)")
                    else:
                        consecutive_failures += 1
                        logger.warning(f"⚠️  Upload failed (consecutive failures: {consecutive_failures})")
                        
                        if consecutive_failures >= max_failures:
                            logger.error(f"❌ Too many consecutive failures ({consecutive_failures}), exiting")
                            break
                
                # Sleep before next check
                time.sleep(CHECK_INTERVAL)
                
            except KeyboardInterrupt:
                logger.info("\n👋 Bot stopped by user")
                break
            except Exception as e:
                logger.error(f"❌ Unexpected error in main loop: {e}")
                consecutive_failures += 1
                
                if consecutive_failures >= max_failures:
                    logger.error(f"❌ Too many consecutive errors, exiting")
                    break
                
                time.sleep(CHECK_INTERVAL)


def main():
    """Entry point"""
    try:
        bot = ConfigUploadBot()
        bot.run()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

