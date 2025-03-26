#!/bin/bash
# ~/MLFlow/scripts/configure_dvc.sh
set -e
if ! command -v dvc >/dev/null 2>&1; then
    echo "DVC not found. Please install it (e.g., 'pip install dvc')."
    exit 1
fi
if ! dvc remote list | grep -q "myremote"; then
    echo "DVC remote 'myremote' not found."
    dvc remote add -d myremote https://dagshub.com/fbarulli/MLFlow.dvc
    echo "DVC remote 'myremote' added."
    read -p "Enter your DagsHub username (e.g., fbarulli): " dvc_user
    read -s -p "Enter your DagsHub token: " dvc_token
    echo
    dvc remote modify myremote --local auth basic
    dvc remote modify myremote --local user "$dvc_user"
    dvc remote modify myremote --local password "$dvc_token"
    echo "DVC remote 'myremote' configured with username: $dvc_user"
else
    echo "DVC remote 'myremote' already configured."
fi
exit 0