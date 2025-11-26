#!/usr/bin/env python3
"""
Abort stuck multipart uploads in R2 bucket.

This script safely aborts incomplete multipart uploads that are stuck in an R2 bucket.
This prevents storage costs from accumulating from incomplete uploads.

Usage:
    python -m distributed_training.scripts.abort_multipart_uploads <bucket_name>
    
    Or as a standalone script:
    python distributed_training/scripts/abort_multipart_uploads.py <bucket_name>

Example:
    python -m distributed_training.scripts.abort_multipart_uploads llama-4b-ws-4-184
"""

import os
import sys
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

import boto3
from dotenv import load_dotenv


def load_env():
    """Load environment variables from .env file."""
    # List of possible .env file locations (in order of priority)
    # Since script is in DistributedTraining root, __file__.parent is the main folder
    possible_env_files = [
        Path(__file__).parent / ".env",  # DistributedTraining main folder (project root)
        Path(__file__).parent / "distributed_training" / ".env",  # distributed_training subfolder
        Path.cwd() / ".env",  # Current working directory
    ]
    
    env_file = None
    for env_path in possible_env_files:
        if env_path.exists():
            env_file = env_path
            break
    
    try:
        if env_file:
            load_dotenv(env_file)
            if os.getenv("R2_ACCOUNT_ID"):  # Verify it loaded
                if os.getenv("DEBUG_ENV"):
                    print(f"✅ Loaded .env from: {env_file}")
                return
        # Fallback to default dotenv behavior
        load_dotenv()
    except ImportError:
        # Fallback: manual .env parsing if dotenv not available
        if env_file and env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and "=" in line and not line.startswith("#"):
                        key, value = line.split("=", 1)
                        os.environ[key.strip()] = value.strip().strip('"').strip("'")


def get_r2_client():
    """Create and return R2 boto3 client with credentials from environment."""
    r2_account_id = os.getenv("R2_ACCOUNT_ID")
    r2_write_key = os.getenv("R2_WRITE_ACCESS_KEY_ID")
    r2_write_secret = os.getenv("R2_WRITE_SECRET_ACCESS_KEY")

    # Validate credentials
    if not all([r2_account_id, r2_write_key, r2_write_secret]):
        print("❌ Missing R2 credentials!")
        print()
        print("Required environment variables:")
        print("  - R2_ACCOUNT_ID")
        print("  - R2_WRITE_ACCESS_KEY_ID")
        print("  - R2_WRITE_SECRET_ACCESS_KEY")
        print()
        print("Please set these in your .env file or environment.")
        sys.exit(1)

    try:
        r2 = boto3.client(
            "s3",
            endpoint_url=f"https://{r2_account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=r2_write_key,
            aws_secret_access_key=r2_write_secret,
            region_name="auto",
        )
        return r2
    except Exception as e:
        print(f"❌ Failed to create R2 client: {e}")
        sys.exit(1)


def abort_all_multipart(r2, bucket_name, age_threshold_hours=1):
    """
    Abort old/stuck multipart uploads in the specified bucket.
    Only aborts uploads older than the specified age threshold to avoid
    interrupting active uploads.
    
    Args:
        r2: boto3 S3 client configured for R2
        bucket_name: Name of the R2 bucket to clean up
        age_threshold_hours: Only abort uploads older than this many hours (default: 1)
    """
    print("=" * 80)
    print("SCANNING FOR STUCK MULTIPART UPLOADS")
    print("=" * 80)
    print()
    print(f"Bucket: {bucket_name}")
    print(f"Age threshold: {age_threshold_hours} hour(s) (only older uploads will be aborted)")
    print()

    total_aborted = 0
    total_skipped = 0
    total_errors = 0
    current_time = datetime.now(timezone.utc)

    try:
        while True:
            resp = r2.list_multipart_uploads(Bucket=bucket_name)
            uploads = resp.get("Uploads", [])

            if not uploads:
                break

            print(f"Found {len(uploads)} multipart upload(s):")
            print()

            for u in uploads:
                key = u["Key"]
                upload_id = u["UploadId"]
                initiated_str = u.get("Initiated", "Unknown")
                
                # Parse the initiated timestamp
                try:
                    if isinstance(initiated_str, str):
                        # Parse ISO format timestamp
                        initiated = datetime.fromisoformat(initiated_str.replace('Z', '+00:00'))
                    else:
                        initiated = initiated_str
                    
                    # Calculate age
                    age_delta = current_time - initiated
                    age_hours = age_delta.total_seconds() / 3600
                    age_str = f"{age_hours:.2f} hours"
                    
                    print(f"  📁 Key: {key}")
                    print(f"     Upload ID: {upload_id[:20]}...")
                    print(f"     Initiated: {initiated_str}")
                    print(f"     Age: {age_str}")
                    
                    # Only abort if older than threshold
                    if age_hours >= age_threshold_hours:
                        print(f"     Status: ⚠️  OLD/STUCK (will abort)")
                        print()
                        try:
                            r2.abort_multipart_upload(
                                Bucket=bucket_name, Key=key, UploadId=upload_id
                            )
                            print(f"  ✅ Aborted: {key}")
                            total_aborted += 1
                        except Exception as e:
                            print(f"  ❌ Failed to abort {key}: {e}")
                            total_errors += 1
                    else:
                        print(f"     Status: ✅ ACTIVE (skipping - less than {age_threshold_hours}h old)")
                        total_skipped += 1
                    print()
                    
                except (ValueError, TypeError) as e:
                    # If we can't parse the timestamp, be conservative and skip it
                    print(f"     Status: ⚠️  UNKNOWN (skipping - cannot parse timestamp)")
                    print(f"     Error: {e}")
                    total_skipped += 1
                    print()

            # Small delay to let R2 finalize the aborts
            time.sleep(0.5)

        print("=" * 80)
        if total_aborted == 0 and total_skipped == 0:
            print("✅ No multipart uploads found!")
        else:
            if total_aborted > 0:
                print(f"✅ SUCCESS: Aborted {total_aborted} old/stuck multipart upload(s)")
            if total_skipped > 0:
                print(f"ℹ️  SKIPPED: {total_skipped} active upload(s) (preserved)")
            if total_errors > 0:
                print(f"⚠️  WARNING: {total_errors} upload(s) failed to abort")
        print("=" * 80)

    except Exception as e:
        print(f"❌ Error listing/aborting multipart uploads: {e}")
        sys.exit(1)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Abort stuck multipart uploads in an R2 bucket",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s llama-4b-ws-4-184
  %(prog)s llama-4b-ws-4-082
        """,
    )
    parser.add_argument(
        "bucket_name",
        type=str,
        help="Name of the R2 bucket to clean up (e.g., llama-4b-ws-4-184)",
    )
    parser.add_argument(
        "--age-threshold",
        "-a",
        type=float,
        default=1.0,
        help="Only abort uploads older than this many hours (default: 1.0). "
             "Active uploads newer than this will be preserved.",
    )
    parser.add_argument(
        "--abort-all",
        action="store_true",
        help="Abort ALL multipart uploads regardless of age (use with caution!)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("ABORT STUCK MULTIPART UPLOADS")
    print("=" * 80)
    print()

    # Load environment variables
    load_env()

    # Get R2 client
    print("Connecting to R2...")
    r2_account_id = os.getenv("R2_ACCOUNT_ID", "")
    if r2_account_id:
        print(f"  Account ID: {r2_account_id[:10]}...")
    print()

    try:
        r2 = get_r2_client()
        print("✅ Connected to R2")
        print()
    except SystemExit:
        # Credentials missing, already printed error
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to connect to R2: {e}")
        sys.exit(1)

    # Abort multipart uploads
    try:
        age_threshold = 0 if args.abort_all else args.age_threshold
        if args.abort_all:
            print("⚠️  WARNING: --abort-all flag set. ALL uploads will be aborted!")
            print()
        abort_all_multipart(r2, args.bucket_name, age_threshold_hours=age_threshold)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

