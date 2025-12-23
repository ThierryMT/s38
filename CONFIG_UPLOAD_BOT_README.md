# Config Upload Bot

## Purpose

This bot uploads your miner's `config.json` file to R2 storage **every 30 seconds** to keep your `block_list` fresh for validator time checking.

## Why This Matters

Validators check your `block_list` (blockchain block numbers when you trained) to ensure your data is fresh:
- **Threshold**: Data must be < 20 minutes old
- **If too old**: Validator throws error: `"Uploaded datatset block older than 20.0 minutes"`
- **Result**: Your assigned score gets reset to 0

## How It Works

1. **Monitors** your local `config.json` file (updated by the miner every batch)
2. **Uploads** to R2 bucket every 30 seconds (or when config changes)
3. **Only uploads config.json** (~5KB) - NOT the full model (~8GB)
4. **Keeps validators happy** - they always see fresh block_list data

## Installation & Usage

### Start the Bot

```bash
cd /workspace/s38
chmod +x start_config_upload_bot.sh
./start_config_upload_bot.sh
```

### Monitor the Bot

```bash
# Check status
pm2 status config_upload_bot

# View logs
pm2 logs config_upload_bot

# View last 50 lines
pm2 logs config_upload_bot --lines 50
```

### Stop/Restart

```bash
# Stop
pm2 stop config_upload_bot

# Restart
pm2 restart config_upload_bot

# Delete
pm2 delete config_upload_bot
```

## Configuration

The bot reads R2 credentials from your `.env` file:
- `R2_ACCOUNT_ID`
- `R2_BUCKET_NAME`
- `R2_WRITE_ACCESS_KEY_ID`
- `R2_WRITE_SECRET_ACCESS_KEY`

### Change Upload Interval

Edit `config_upload_bot.py`:

```python
UPLOAD_INTERVAL = 30  # Change to desired seconds
```

**Recommendations:**
- **30 seconds**: Safe, frequent updates (default)
- **60 seconds**: Less frequent but still safe
- **< 30 seconds**: Not recommended (unnecessary overhead)

## What Gets Uploaded

Only the `config.json` file containing:
- `block_list`: Array of blockchain block numbers
- Model configuration metadata
- Training progress info

**Size**: ~5KB per upload

## Benefits vs Regular Upload

| Aspect | Regular Miner Upload | Config Upload Bot |
|--------|---------------------|-------------------|
| **Frequency** | Every 60 training steps | Every 30 seconds |
| **Size** | ~8GB (full model) | ~5KB (config only) |
| **Upload Time** | 3-5 minutes | < 1 second |
| **Network Usage** | High | Minimal |
| **Validator Check** | May timeout | Always fresh |

## Logs

Bot logs are stored in:
- `logs/config_upload_bot_out.log` - Standard output
- `logs/config_upload_bot_error.log` - Errors only
- `logs/config_upload_bot_combined.log` - Combined

## Troubleshooting

### Bot Not Starting

```bash
# Check if file exists and is executable
ls -la config_upload_bot.py
chmod +x config_upload_bot.py

# Check if R2 credentials are set
cat .env | grep R2
```

### Uploads Failing

```bash
# View error logs
pm2 logs config_upload_bot --err --lines 50

# Common issues:
# - R2 credentials not set
# - Bucket name incorrect
# - Network connectivity issues
```

### Config.json Not Found

The bot will wait for the miner to create `config.json`. Make sure:
- The miner is running
- The miner has completed initialization
- The output directory is correct

## Expected Log Output

```
2025-12-22 12:00:00 - INFO - 🚀 Config Upload Bot Initialized
2025-12-22 12:00:00 - INFO - 📦 Bucket: llama-4b-ws-4-227
2025-12-22 12:00:00 - INFO - 📁 Output Dir: /workspace/s38/llama-4b-ws-4-227
2025-12-22 12:00:00 - INFO - ⏱️  Upload Interval: 30 seconds
2025-12-22 12:00:00 - INFO - 🔄 Starting config upload loop...
2025-12-22 12:00:30 - INFO - 🔔 Upload triggered: time interval reached
2025-12-22 12:00:30 - INFO - 📤 Uploading config.json (15 blocks, latest: [1234, 1235, 1236])...
2025-12-22 12:00:31 - INFO - ✅ Successfully uploaded config.json
2025-12-22 12:00:31 - INFO - ⏳ Next upload at 12:01:01 (or when config changes)
```

## Integration with Main Miner

This bot runs **independently** from your main miner:
- Miner continues normal uploads every 60 steps (full model)
- Bot uploads just config.json every 30 seconds
- Both work together to maximize your score

## When to Use

✅ **Use this bot if:**
- You're getting "datatset block older than X minutes" errors
- Your training steps are slow
- You want to maximize assigned score
- You have stable network connectivity

❌ **Don't need this bot if:**
- Your miner uploads frequently enough already
- You have very fast training (< 30 seconds per 60 steps)
- You're not participating in scoring

