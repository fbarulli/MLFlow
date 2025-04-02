#!/bin/bash
# ~/MLFlow/scripts/configure_dvc.sh
set -e
cd "$(dirname "$(dirname "$0")")"
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
# Create .dvc/config if it doesn't exist
mkdir -p .dvc
touch .dvc/config

# Write DVC configuration directly
cat > .dvc/config << EOF
[core]
    autostage = false
[remote "origin"]
    url = ${EXPECTED_URL}
    auth = basic
    user = fbarulli
    password = 1de275ede522e8bd56e558a81ecd32a803b7ba64
EOF

# Set origin as the default remote
echo "Setting 'origin' as default DVC remote..."
dvc remote default origin

# Verify configuration
echo "Verifying DVC configuration..."
if ! dvc remote list | grep -q "origin"; then
    echo "Error: DVC remote 'origin' not configured"
    exit 1
fi

if ! dvc status; then
    echo "Error: DVC status check failed"
    exit 1
fi
echo "DVC configured with remote: $EXPECTED_URL"
echo "Default remote set to: origin"