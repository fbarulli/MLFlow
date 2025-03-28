#!/bin/bash
# ~/MLFlow/scripts/configure_dvc.sh
set -e

# Ensure working directory is correct
cd /home/ubuntu/MLFlow

if ! command -v dvc &> /dev/null; then
    echo "DVC not installed. Please install it first."
    exit 1
fi

# Expected remote URL
EXPECTED_URL="https://dagshub.com/fbarulli/MLFlow.dvc"

# Check existing remote
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

# Set credentials
echo "Setting DVC credentials..."
dvc remote modify origin --local auth basic
dvc remote modify origin --local user fbarulli
dvc remote modify origin --local password 338c74d36ff47a81dd766e2c0de58b72ef9de932
dvc config --global core.autostage false

echo "DVC configured with remote: $EXPECTED_URL"