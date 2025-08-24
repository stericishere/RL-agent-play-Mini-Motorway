#!/bin/bash

# Mini Motorways RL Player Demo Script

set -e

echo "🎮 Mini Motorways RL Player Demo"
echo "================================="

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This demo requires macOS"
    exit 1
fi

# Check if motorways command is available
if ! command -v motorways &> /dev/null; then
    echo "❌ 'motorways' command not found"
    echo "Please install the package first:"
    echo "  pip install -e ."
    exit 1
fi

echo ""
echo "📋 Pre-Demo Checklist:"
echo "----------------------"
echo "✓ Make sure Mini Motorways is running and visible"
echo "✓ Grant Screen Recording permission to your Terminal"
echo "✓ Grant Accessibility permission to your Terminal"
echo "✓ Position Mini Motorways window where you can see it"
echo ""

read -p "Press Enter when ready to continue..."

echo ""
echo "🔧 Step 1: Check Permissions"
echo "-----------------------------"

# Basic permission check by trying to run help
motorways --help > /dev/null

echo "✓ Package installed and accessible"

echo ""
echo "🎯 Step 2: Grid Calibration"
echo "---------------------------"
echo "This will guide you through calibrating the game grid."
echo ""

# Check if calibration already exists
CALIBRATION_FILE="$HOME/.motorways/calibration.json"

if [[ -f "$CALIBRATION_FILE" ]]; then
    echo "📁 Found existing calibration at $CALIBRATION_FILE"
    read -p "Do you want to recalibrate? (y/N): " recalibrate
    
    if [[ "$recalibrate" =~ ^[Yy]$ ]]; then
        echo "Running new calibration..."
        motorways calibrate --grid-h 32 --grid-w 32
    else
        echo "Using existing calibration."
    fi
else
    echo "No existing calibration found. Running calibration..."
    motorways calibrate --grid-h 32 --grid-w 32
fi

echo ""
echo "✅ Calibration complete!"

echo ""
echo "🧪 Step 3: Dry Run Test"
echo "-----------------------"
echo "Testing setup with random actions (no actual clicks)"
echo ""

motorways dry-run --max-steps 10 --fps 2

echo ""
echo "✅ Dry run complete!"

echo ""
echo "🎮 Step 4: Demo Options"
echo "----------------------"
echo "Choose what to do next:"
echo ""
echo "1) Test with more dry-run steps (safe, no clicks)"
echo "2) Load your own RL model and play live"
echo "3) Exit demo"
echo ""

while true; do
    read -p "Enter choice (1/2/3): " choice
    case $choice in
        1)
            echo ""
            echo "Running extended dry run..."
            motorways dry-run --max-steps 50 --fps 4
            break
            ;;
        2)
            echo ""
            echo "🤖 Live Model Play"
            echo "Enter the path to your trained model file:"
            echo "(Supported formats: .zip for SB3, .pt/.pth for PyTorch)"
            echo ""
            
            while true; do
                read -p "Model path: " model_path
                
                if [[ -z "$model_path" ]]; then
                    echo "❌ Please enter a model path"
                    continue
                fi
                
                if [[ ! -f "$model_path" ]]; then
                    echo "❌ File not found: $model_path"
                    continue
                fi
                
                echo ""
                echo "🚀 Starting live agent play..."
                echo "⚠️  The agent will perform real mouse clicks!"
                echo "⚠️  Move mouse to screen corner to trigger failsafe if needed"
                echo ""
                
                read -p "Press Enter to start, or Ctrl+C to cancel..."
                
                motorways play --model "$model_path" --max-steps 100 --fps 6
                break
            done
            break
            ;;
        3)
            echo "👋 Exiting demo"
            exit 0
            ;;
        *)
            echo "❌ Invalid choice. Please enter 1, 2, or 3."
            ;;
    esac
done

echo ""
echo "🎉 Demo Complete!"
echo "================="
echo ""
echo "📝 What happened:"
echo "- Calibrated your Mini Motorways window grid"
echo "- Tested screen capture and coordinate mapping"
echo "- Demonstrated the agent control system"
echo ""
echo "📚 Next Steps:"
echo "- Check logs in: ~/.motorways/logs/"
echo "- Modify settings in: ~/.motorways/calibration.json"
echo "- Train your own RL model for better performance"
echo "- Read the README.md for full documentation"
echo ""
echo "🆘 Need Help?"
echo "- Run: motorways --help"
echo "- Check permissions in System Settings"
echo "- Review troubleshooting in README.md"
echo ""
echo "Happy reinforcement learning! 🤖✨"