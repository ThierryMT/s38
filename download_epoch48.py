import boto3
import os

# Load environment variables from s38/.env
env_file = '/workspace/s38/.env'
env_vars = {}
with open(env_file, 'r') as f:
    for line in f:
        if line.strip() and not line.startswith('#'):
            # Remove 'export ' if present
            line = line.strip().replace('export ', '')
            if '=' in line:
                key, value = line.split('=', 1)
                env_vars[key] = value

# Get R2 credentials
account_id = env_vars['R2_ACCOUNT_ID']
bucket_name = env_vars['R2_BUCKET_NAME']
access_key = env_vars['R2_READ_ACCESS_KEY_ID']
secret_key = env_vars['R2_READ_SECRET_ACCESS_KEY']

print("=" * 80)
print("DOWNLOADING EPOCH-48 FILES FROM R2")
print("=" * 80)
print(f"Bucket: {bucket_name}")
print(f"Account: {account_id[:10]}...")
print(f"Destination: /workspace/model/epoch-48/")
print()

# Create R2 client
r2 = boto3.client(
    's3',
    endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
    aws_access_key_id=access_key,
    aws_secret_access_key=secret_key,
    region_name='auto'
)

# List all objects with epoch-48 prefix
print("Scanning for epoch-48 files...")
response = r2.list_objects_v2(Bucket=bucket_name, Prefix='epoch-48/')

if 'Contents' not in response:
    print("❌ No files found in epoch-48/")
    exit(1)

files = response['Contents']
print(f"✅ Found {len(files)} file(s)\n")

# Download each file
downloaded = 0
failed = 0

for obj in files:
    key = obj['Key']
    size = obj['Size']
    size_mb = size / (1024 * 1024)
    
    # Skip if it's just the directory marker
    if key.endswith('/'):
        continue
    
    # Create local path
    local_path = f"/workspace/model/{key}"
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    try:
        print(f"📥 Downloading: {key} ({size_mb:.2f} MB)...")
        r2.download_file(bucket_name, key, local_path)
        print(f"   ✅ Saved to: {local_path}")
        downloaded += 1
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        failed += 1
    print()

print("=" * 80)
print(f"✅ SUCCESS: Downloaded {downloaded} file(s)")
if failed > 0:
    print(f"⚠️  Failed: {failed} file(s)")
print("=" * 80)
print(f"\nFiles saved to: /workspace/model/epoch-48/")
