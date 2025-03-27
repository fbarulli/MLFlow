#!/bin/bash
# ~/MLFlow/scripts/configure_dvc.sh
set -e

if ! command -v dvc &> /dev/null; then
    echo "DVC not installed. Please install it first."
    exit 1
fi

DVC_REMOTE_URL=$(dvc remote list 2>/dev/null | grep "origin" | awk '{print $2}' || true)

if [ -z "$DVC_REMOTE_URL" ]; then
    echo "Configuring DVC with hardcoded remote..."
    dvc remote add origin https://dagshub.com/fbarulli/MLFlow
    dvc remote modify origin --local auth basic
    dvc remote modify origin --local user fbarulli
    dvc remote modify origin --local password 338c74d36ff47a81dd766e2c0de58b72ef9de932
    dvc config --global core.autostage false
    echo "DVC configured with remote: https://dagshub.com/fbarulli/MLFlow"
else
    if [ "$DVC_REMOTE_URL" = "https://dagshub.com/fbarulli/MLFlow" ]; then
        echo "DVC remote already set to expected value: $DVC_REMOTE_URL"
        dvc config --global core.autostage false
    else
        echo "DVC remote exists but does not match expected value: $DVC_REMOTE_URL"
        echo "Expected: https://dagshub.com/fbarulli/MLFlow"
        exit 1
    fi
fi