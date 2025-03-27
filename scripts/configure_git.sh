#!/bin/bash
# ~/MLFlow/scripts/configure_git.sh
# Remove set -e to prevent early termination on commands that might fail
# set -e

echo "Starting Git configuration..."

# Check if git is installed and accessible
if ! command -v git &> /dev/null; then
    echo "Error: Git is not installed or not in PATH"
    exit 1
fi

echo "Git is installed at: $(which git)"
echo "Git version: $(git --version)"

# Use safer command execution with explicit error handling
GIT_EMAIL=$(git config --global user.email || echo "")
echo "Debug: Git email = '$GIT_EMAIL'"

GIT_NAME=$(git config --global user.name || echo "")
echo "Debug: Git name = '$GIT_NAME'"

if [ -z "$GIT_EMAIL" ] || [ -z "$GIT_NAME" ]; then
    if [ -t 0 ]; then
        echo "Git user configuration not found."
        read -p "Enter your Git email: " git_email
        read -p "Enter your Git name: " git_name
        git config --global user.email "$git_email"
        git config --global user.name "$git_name"
        echo "Git configured with email: $git_email, name: $git_name"
    else
        echo "Git configuration not found, but running non-interactively."
        echo "Setting default Git configuration..."
        # Set default values when running non-interactively
        git config --global user.email "mlflow-user@example.com"
        git config --global user.name "MLflow User"
        echo "Git configured with default values"
    fi
else
    echo "Git configuration already set: email=$GIT_EMAIL, name=$GIT_NAME"
fi

echo "Git configuration completed successfully."
exit 0