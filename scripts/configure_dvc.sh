#!/bin/bash
# ~/MLFlow/scripts/configure_dvc.sh
set -e
cd /home/ubuntu/MLFlow
if ! command -v dvc &> /dev/null; then
    echo "DVC not installed. Please install it first."
    exit 1
fi
EXPECTED_URL="https://dagshub.com/fbarulli/MLFlow.dvc"
DVC_REMOTE_URL=$(dvc remote list 2>/dev/null | grep "origin" | awk '{print $2}' || true)
if [ -z "$DVC_REMOTE_URL" ]; then
    echo "Configuring DVC with hardcoded remote..."
    dvc remote add origin "$EXPECTED_URL"
elif [ "$DVC_REMOTE_URL" != "$EXPECTED_URL" ]; then
    echo "DVC remote exists but does not match expected value: $DVC_REMOTE_URL"
    echo "Updating to expected: $EXPECTED_URL"
    dvc remote modify origin url "$EXPECTED_URL"
else
    echo "DVC remote already set to expected value: $DVC_REMOTE_URL"
fi
echo "Setting DVC credentials..."
dvc remote modify origin --local auth basic
dvc remote modify origin --local user fbarulli
dvc remote modify origin --local password <your-new-dagshub-token>
dvc config --global core.autostage false
echo "DVC configured with remote: $EXPECTED_URL"