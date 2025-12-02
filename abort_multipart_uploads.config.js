// PM2 configuration for abort_multipart_uploads.py
module.exports = {
  apps: [{
    name: 'abort_multipart_uploads',
    script: './abort_multipart_uploads.py',
    interpreter: 'python3',
    // Replace 'YOUR_BUCKET_NAME' with your actual bucket name
    // Examples: 'llama-4b-ws-4-184', 'llama-4b-ws-4-082'
    args: ['llama-4b-ws-4-227'],
    // Optional: Add additional arguments
    // args: ['YOUR_BUCKET_NAME', '--age-threshold', '2.0'],
    // args: ['YOUR_BUCKET_NAME', '--abort-all'],  // Use with caution!
    // args: ['YOUR_BUCKET_NAME', '--verbose'],
    autorestart: false,  // Set to false if you want it to run once and exit
    watch: false,
    max_memory_restart: '500M',
    error_file: './logs/abort_multipart_uploads_error.log',
    out_file: './logs/abort_multipart_uploads_out.log',
    log_file: './logs/abort_multipart_uploads_combined.log',
    time: true,
    env: {
      NODE_ENV: 'production'
    }
  }]
};

