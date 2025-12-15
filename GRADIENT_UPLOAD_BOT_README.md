# Gradient Upload Bot

A bot that automatically uploads `gradients.pt` every 3 minutes to keep your repo score fresh. It checks the current epoch before uploading to ensure the gradient is uploaded to the correct epoch folder.

## Features

- ✅ Uploads `gradients.pt` every 3 minutes
- ✅ Checks current epoch from `metadata.json` before uploading
- ✅ Uploads to correct epoch folder: `epoch-{epoch}/gradients.pt`
- ✅ Validates gradient file before uploading
- ✅ Also uploads `metadata.json` to keep it in sync
- ✅ Automatic retry on failures
- ✅ Comprehensive logging

## Requirements

1. **R2 Credentials** (in `.env` file or environment):
   - `R2_ACCOUNT_ID`
   - `R2_WRITE_ACCESS_KEY_ID`
   - `R2_WRITE_SECRET_ACCESS_KEY`
   - `R2_BUCKET_NAME` (optional, will auto-detect from output directory)

2. **Miner Running**: The miner must be running and saving `gradients.pt` to the output directory

## How It Works

1. **Reads Current Epoch**: Reads `metadata.json` from the output directory to get the current epoch
2. **Validates Gradient File**: Checks if `gradients.pt` exists and is valid
3. **Uploads to Correct Folder**: Uploads to `epoch-{epoch}/gradients.pt` in your R2 bucket
4. **Keeps Metadata in Sync**: Also uploads `metadata.json` to the same epoch folder

## Usage

### Option 1: Run with PM2 (Recommended)

```bash
# Start the bot
pm2 start gradient_upload_bot.config.js

# Check status
pm2 status

# View logs
pm2 logs gradient_upload_bot

# Stop the bot
pm2 stop gradient_upload_bot
```

### Option 2: Run Directly

```bash
# Make sure you're in the s38 directory
cd /workspace/s38

# Run the bot
python3 gradient_upload_bot.py
```

## Configuration

The bot automatically detects:
- **Bucket Name**: From output directory (e.g., `llama-4b-ws-4-227`)
- **Output Directory**: Where `gradients.pt` is saved (usually `{bucket_name}/`)

You can override by setting `R2_BUCKET_NAME` in your `.env` file.

## Upload Interval

Default: **3 minutes (180 seconds)**

To change, edit `UPLOAD_INTERVAL` in `gradient_upload_bot.py`:

```python
UPLOAD_INTERVAL = 180  # Change to your desired interval in seconds
```

## Logs

Logs are written to:
- **File**: `gradient_upload_bot.log`
- **PM2 Logs**: `./logs/gradient_upload_bot_out.log` and `./logs/gradient_upload_bot_error.log`
- **Console**: stdout

## Example Output

```
2025-12-15 15:20:00 - INFO - ======================================================================
2025-12-15 15:20:00 - INFO - 🚀 Gradient Upload Bot Initialized
2025-12-15 15:20:00 - INFO - 📦 Bucket: llama-4b-ws-4-227
2025-12-15 15:20:00 - INFO - 📁 Output Dir: /workspace/s38/llama-4b-ws-4-227
2025-12-15 15:20:00 - INFO - ⏱️  Upload Interval: 180 seconds (3 minutes)
2025-12-15 15:20:00 - INFO - ======================================================================
2025-12-15 15:20:00 - INFO - 🔄 Starting gradient upload loop...
2025-12-15 15:20:00 - INFO - 📤 Uploading gradients.pt to epoch-5/gradients.pt...
2025-12-15 15:20:05 - INFO - ✅ Successfully uploaded gradients.pt to epoch-5/gradients.pt
2025-12-15 15:20:05 - INFO - ✅ Upload successful! Epoch: 5, Inner Step: 120
2025-12-15 15:20:05 - INFO - ⏳ Next upload in 180 seconds...
```

## Troubleshooting

### Bot can't find bucket name
- Set `R2_BUCKET_NAME` in your `.env` file
- Or ensure the output directory exists (e.g., `llama-4b-ws-4-227/`)

### Gradient file not found
- Make sure the miner is running and has saved `gradients.pt`
- Check that the output directory path is correct

### Upload failures
- Check R2 credentials in `.env` file
- Verify R2 write permissions
- Check network connectivity

### Epoch mismatch warnings
- This is normal if the gradient was saved before an epoch change
- The bot will still upload, but the gradient may be from the previous epoch

## Notes

- The bot uploads the same `gradients.pt` file that gets uploaded with the full model
- This ensures consistency between frequent gradient uploads and full model uploads
- The gradient file is validated before upload to ensure it contains valid data
- If validation fails, the bot will still attempt to upload (with a warning)

## Integration with Full Model Uploads

The bot works alongside the miner's full model upload process:
- **Bot**: Uploads `gradients.pt` every 3 minutes (fast, keeps repo score fresh)
- **Miner**: Uploads full model (model.safetensors, config.json, etc.) every `upload_steps`

Both use the same `gradients.pt` file, ensuring consistency.

