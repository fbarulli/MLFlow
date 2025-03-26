#!/bin/bash
# ~/MLFlow/scripts/configure_git.sh
set -e

GIT_EMAIL=$(git config --global user.email)
GIT_NAME=$(git config --global user.name)

if [ -z "$GIT_EMAIL" ] || [ -z "$GIT_NAME" ]; then
    echo "Git user configuration not found."
    read -p "Enter your Git email: " git_email
    read -p "Enter your Git name: " git_name
    git config --global user.email "$git_email"
    git config --global user.name "$git_name"
    echo "Git configured with email: $git_email, name: $git_name"
else
    echo "Git configuration already set: email=$GIT_EMAIL, name=$GIT_NAME"
fi