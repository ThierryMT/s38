#!/usr/bin/env python3
"""
Gradient Upload Bot
Uploads gradients.pt every 3 minutes to keep repo score fresh.
Checks current epoch before uploading to ensure correct epoch folder.
"""

import os
import sys
import time
import json
import logging
import boto3
from pathlib import Path
from datetime import datetime
from botocore.config import Config
from botocore.exceptions import ClientError, NoCredentialsError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gradient_upload_bot.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_INTERVAL = 180  # 3 minutes in seconds
GRADIENT_FILE = "gradients.pt"
METADATA_FILE = "metadata.json"

class GradientUploadBot:
    def __init__(self):
        """Initialize the gradient upload bot"""
        # Get R2 credentials from environment
        self.r2_account_id = os.getenv("R2_ACCOUNT_ID")
        self.r2_write_access_key_id = os.getenv("R2_WRITE_ACCESS_KEY_ID")
        self.r2_write_secret_access_key = os.getenv("R2_WRITE_SECRET_ACCESS_KEY")
        
        # Get bucket name - try to infer from output_dir or use env var
        self.bucket_name = os.getenv("R2_BUCKET_NAME")
        if not self.bucket_name:
            # Try to find bucket name from output_dir
            self.bucket_name = self._find_bucket_name()
        
        # Get output directory (where gradients.pt is saved)
        self.output_dir = self._find_output_dir()
        
        # Validate configuration
        self._validate_config()
        
        # Initialize R2 client
        self.r2_client = self._init_r2_client()
        
        logger.info("=" * 70)
        logger.info("🚀 Gradient Upload Bot Initialized")
        logger.info(f"📦 Bucket: {self.bucket_name}")
        logger.info(f"📁 Output Dir: {self.output_dir}")
        logger.info(f"⏱️  Upload Interval: {UPLOAD_INTERVAL} seconds (3 minutes)")
        logger.info("=" * 70)
    
    def _find_bucket_name(self):
        """Try to find bucket name from output directory"""
        # Look for directories matching pattern: llama-4b-ws-4-XXX
        current_dir = Path.cwd()
        for item in current_dir.iterdir():
            if item.is_dir() and item.name.startswith("llama-4b-ws-4-"):
                logger.info(f"📦 Found bucket name from directory: {item.name}")
                return item.name
        return None
    
    def _find_output_dir(self):
        """Find the output directory where gradients.pt is saved"""
        if self.bucket_name:
            output_dir = Path.cwd() / self.bucket_name
            if output_dir.exists():
                return str(output_dir)
        
        # Try to find any directory with gradients.pt
        current_dir = Path.cwd()
        for item in current_dir.iterdir():
            if item.is_dir():
                gradient_path = item / GRADIENT_FILE
                if gradient_path.exists():
                    logger.info(f"📁 Found output dir: {item}")
                    return str(item)
        
        # Default to current directory
        return str(current_dir)
    
    def _validate_config(self):
        """Validate that all required configuration is present"""
        errors = []
        
        if not self.r2_account_id:
            errors.append("R2_ACCOUNT_ID environment variable not set")
        if not self.r2_write_access_key_id:
            errors.append("R2_WRITE_ACCESS_KEY_ID environment variable not set")
        if not self.r2_write_secret_access_key:
            errors.append("R2_WRITE_SECRET_ACCESS_KEY environment variable not set")
        if not self.bucket_name:
            errors.append("R2_BUCKET_NAME not found (checked env var and output directories)")
        
        if errors:
            logger.error("❌ Configuration Errors:")
            for error in errors:
                logger.error(f"   - {error}")
            raise ValueError("Missing required configuration. Please check your .env file.")
    
    def _init_r2_client(self):
        """Initialize R2 S3 client"""
        try:
            client = boto3.client(
                "s3",
                endpoint_url=f"https://{self.r2_account_id}.r2.cloudflarestorage.com",
                aws_access_key_id=self.r2_write_access_key_id,
                aws_secret_access_key=self.r2_write_secret_access_key,
                region_name="auto",
                config=Config(
                    retries={"max_attempts": 10, "mode": "adaptive"},
                    connect_timeout=30,
                    read_timeout=120,
                    max_pool_connections=50,
                ),
            )
            logger.info("✅ R2 client initialized successfully")
            return client
        except Exception as e:
            logger.error(f"❌ Failed to initialize R2 client: {e}")
            raise
    
    def get_current_epoch(self):
        """Read current epoch from metadata.json"""
        metadata_path = Path(self.output_dir) / METADATA_FILE
        
        if not metadata_path.exists():
            logger.warning(f"⚠️  metadata.json not found at {metadata_path}")
            return None
        
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            epoch = metadata.get("outer_step")
            inner_step = metadata.get("inner_step", 0)
            
            if epoch is None:
                logger.warning("⚠️  'outer_step' not found in metadata.json")
                return None
            
            logger.debug(f"📊 Current epoch: {epoch}, inner_step: {inner_step}")
            return epoch, inner_step
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse metadata.json: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error reading metadata.json: {e}")
            return None
    
    def check_gradient_file(self, expected_epoch=None):
        """Check if gradients.pt exists and is valid, and matches current epoch"""
        gradient_path = Path(self.output_dir) / GRADIENT_FILE
        
        if not gradient_path.exists():
            logger.warning(f"⚠️  {GRADIENT_FILE} not found at {gradient_path}")
            return False
        
        # Check file size (should be > 0)
        file_size = gradient_path.stat().st_size
        if file_size == 0:
            logger.warning(f"⚠️  {GRADIENT_FILE} is empty (0 bytes)")
            return False
        
        # Check file modification time (should be recent)
        mod_time = gradient_path.stat().st_mtime
        age_seconds = time.time() - mod_time
        
        if age_seconds > 600:  # Older than 10 minutes
            logger.warning(f"⚠️  {GRADIENT_FILE} is {age_seconds/60:.1f} minutes old")
        
        # Verify gradient file contains valid data (basic check)
        try:
            import torch
            gradient_data = torch.load(str(gradient_path), weights_only=True, map_location="cpu")
            if not isinstance(gradient_data, dict):
                logger.warning(f"⚠️  {GRADIENT_FILE} does not contain a dictionary")
                return False
            
            # Check if gradient has expected keys (should have model parameter keys)
            if len(gradient_data) == 0:
                logger.warning(f"⚠️  {GRADIENT_FILE} is empty (no parameters)")
                return False
            
            # Check if metadata exists and matches epoch
            if "metadata" in gradient_data:
                metadata = gradient_data.get("metadata", {})
                if expected_epoch is not None:
                    grad_epoch = metadata.get("outer_step")
                    if grad_epoch is not None and grad_epoch != expected_epoch:
                        logger.warning(f"⚠️  Gradient epoch mismatch: file has {grad_epoch}, expected {expected_epoch}")
                        # Still allow upload, but warn
            
            logger.debug(f"✅ {GRADIENT_FILE} is valid ({file_size / 1024 / 1024:.2f} MB, {age_seconds:.1f}s old, {len(gradient_data)} parameters)")
            return True
        except ImportError:
            # torch not available, skip validation but still allow upload
            logger.debug(f"⚠️  torch not available, skipping gradient validation")
            logger.info(f"✅ {GRADIENT_FILE} exists ({file_size / 1024 / 1024:.2f} MB, {age_seconds:.1f}s old)")
            return True
        except Exception as e:
            logger.warning(f"⚠️  Failed to validate {GRADIENT_FILE}: {e} (continuing anyway)")
            # Still allow upload even if validation fails
            logger.info(f"✅ {GRADIENT_FILE} exists ({file_size / 1024 / 1024:.2f} MB, {age_seconds:.1f}s old)")
            return True
    
    def upload_gradient(self, epoch):
        """Upload gradients.pt to the correct epoch folder"""
        gradient_path = Path(self.output_dir) / GRADIENT_FILE
        metadata_path = Path(self.output_dir) / METADATA_FILE
        
        if not gradient_path.exists():
            logger.error(f"❌ Cannot upload: {GRADIENT_FILE} not found")
            return False
        
        # Construct upload paths
        prefix = f"epoch-{epoch}/"
        gradient_key = f"{prefix}{GRADIENT_FILE}"
        metadata_key = f"{prefix}{METADATA_FILE}"
        
        try:
            # Upload gradients.pt
            logger.info(f"📤 Uploading {GRADIENT_FILE} to {gradient_key}...")
            # Use ExtraArgs for multipart configuration if needed
            # For large files, boto3 will automatically use multipart upload
            self.r2_client.upload_file(
                str(gradient_path),
                self.bucket_name,
                gradient_key,
            )
            logger.info(f"✅ Successfully uploaded {GRADIENT_FILE} to {gradient_key}")
            
            # Also upload metadata.json to keep it in sync
            if metadata_path.exists():
                logger.info(f"📤 Uploading {METADATA_FILE} to {metadata_key}...")
                self.r2_client.upload_file(
                    str(metadata_path),
                    self.bucket_name,
                    metadata_key,
                )
                logger.info(f"✅ Successfully uploaded {METADATA_FILE} to {metadata_key}")
            
            return True
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            logger.error(f"❌ R2 upload failed ({error_code}): {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error during upload: {e}")
            return False
    
    def run(self):
        """Main loop: upload gradients every 3 minutes"""
        logger.info("🔄 Starting gradient upload loop...")
        last_epoch = None
        consecutive_failures = 0
        max_failures = 5
        
        while True:
            try:
                # Get current epoch
                epoch_info = self.get_current_epoch()
                if epoch_info is None:
                    logger.warning("⚠️  Could not get current epoch, skipping this cycle")
                    time.sleep(UPLOAD_INTERVAL)
                    continue
                
                epoch, inner_step = epoch_info
                
                # Check if epoch changed
                if last_epoch is not None and epoch != last_epoch:
                    logger.info(f"🔄 Epoch changed: {last_epoch} → {epoch}")
                
                # Check if gradient file exists and is valid for current epoch
                if not self.check_gradient_file(expected_epoch=epoch):
                    logger.warning("⚠️  Gradient file check failed, skipping upload")
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        logger.error(f"❌ Too many consecutive failures ({consecutive_failures}), exiting")
                        break
                    time.sleep(UPLOAD_INTERVAL)
                    continue
                
                # Upload gradient
                success = self.upload_gradient(epoch)
                
                if success:
                    consecutive_failures = 0
                    logger.info(f"✅ Upload successful! Epoch: {epoch}, Inner Step: {inner_step}")
                    logger.info(f"⏳ Next upload in {UPLOAD_INTERVAL} seconds...")
                else:
                    consecutive_failures += 1
                    logger.warning(f"⚠️  Upload failed (consecutive failures: {consecutive_failures})")
                    if consecutive_failures >= max_failures:
                        logger.error(f"❌ Too many consecutive failures ({consecutive_failures}), exiting")
                        break
                
                last_epoch = epoch
                
            except KeyboardInterrupt:
                logger.info("🛑 Received interrupt signal, shutting down...")
                break
            except Exception as e:
                logger.error(f"❌ Unexpected error in main loop: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_failures:
                    logger.error(f"❌ Too many consecutive failures ({consecutive_failures}), exiting")
                    break
            
            # Wait for next upload cycle
            time.sleep(UPLOAD_INTERVAL)


def main():
    """Main entry point"""
    try:
        bot = GradientUploadBot()
        bot.run()
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

