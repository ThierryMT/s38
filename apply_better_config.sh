#!/bin/bash
echo "Backing up current config..."
cp app.config.js app.config.backup_$(date +%Y%m%d_%H%M%S).js

echo "Applying OPTIMAL configuration..."
# Option 2 (your proposed)
# sed -i "s/'--neuron.upload_steps','120'/'--neuron.upload_steps','60'/g" app.config.js
# sed -i "s/'--neuron.local_batch_size_train_effective','16'/'--neuron.local_batch_size_train_effective','32'/g" app.config.js

# Option 3 (recommended - much better)
sed -i "s/'--neuron.upload_steps','120'/'--neuron.upload_steps','60'/g" app.config.js
sed -i "s/'--neuron.local_batch_size_train_effective','16'/'--neuron.local_batch_size_train_effective','512'/g" app.config.js

echo "✅ Config updated!"
echo ""
echo "Changes applied:"
grep "upload_steps\|local_batch_size_train_effective" app.config.js
echo ""
echo "Next: pm2 restart distributed_training_miner --update-env"
