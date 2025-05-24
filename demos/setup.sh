#!/bin/bash

# Setup script for Active Inference CartPole Demo
# This script sets up the environment and runs the demo

set -e  # Exit on any error

echo "Setting up Active Inference CartPole Demo..."

# Check if we're in the demos directory
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Please run this script from the demos directory"
    exit 1
fi

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "Poetry not found. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    export PATH="$HOME/.local/bin:$PATH"
fi

# Check if UV is available (optional)
UV_AVAILABLE=false
if command -v uv &> /dev/null; then
    echo "UV detected - using UV for faster dependency management"
    UV_AVAILABLE=true
fi

# Install dependencies
echo "Installing dependencies..."
if [ "$UV_AVAILABLE" = true ]; then
    echo "Using UV for installation..."
    uv sync
else
    echo "Using Poetry for installation..."
    poetry install
fi

echo "Setup complete!"
echo ""
echo "Available commands:"
echo "  # Run default demo:"
if [ "$UV_AVAILABLE" = true ]; then
    echo "  uv run python -m active_inference_cartpole.main"
else
    echo "  poetry run python -m active_inference_cartpole.main"
fi
echo ""
echo "  # Compare agents:"
if [ "$UV_AVAILABLE" = true ]; then
    echo "  uv run python -m active_inference_cartpole.main --mode compare"
else
    echo "  poetry run python -m active_inference_cartpole.main --mode compare"
fi
echo ""
echo "  # Run with visualization:"
if [ "$UV_AVAILABLE" = true ]; then
    echo "  uv run python -m active_inference_cartpole.main --mode demo --render"
else
    echo "  poetry run python -m active_inference_cartpole.main --mode demo --render"
fi
echo ""

# Offer to run the demo
read -p "Would you like to run the demo now? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running demo..."
    if [ "$UV_AVAILABLE" = true ]; then
        uv run python -m active_inference_cartpole.main --mode demo --episodes 50
    else
        poetry run python -m active_inference_cartpole.main --mode demo --episodes 50
    fi
fi