#!/bin/bash

BLUE='\033[0;34m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

# Check if Python3 is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}Python3 not found, trying to install...${NC}"
    
    # try to install Python3
    if command -v apt-get &> /dev/null; then
        sudo apt-get update && sudo apt-get install -y python3 python3-pip python3-venv || { echo -e "${RED}Failed to install Python3 ${NC}"; exit 1; }
    elif command -v yum &> /dev/null; then
        sudo yum install -y python3 python3-pip || { echo -e "${RED}Failed to install Python3 ${NC}"; exit 1; }
    else
        echo -e "${RED}Failed to install Python3, please install it manually and try again${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}Python3 installed successfully!${NC}"
fi

# Create and activate virtual environment (if it doesn't exist)
if [ ! -d "venv" ]; then
    echo -e "${BLUE}Creating Python virtual environment...${NC}"
    python3 -m venv venv || { echo -e "${RED}Failed to create virtual environment${NC}"; exit 1; }
fi

echo -e "${BLUE}Activating virtual environment...${NC}"
source venv/bin/activate || { echo -e "${RED}Failed to activate virtual environment${NC}"; exit 1; }

# Install backend dependencies
#echo -e "${BLUE}Installing backend dependencies...${NC}"
#pip3 install -r requirements.txt || { echo -e "${RED}Failed to install dependencies${NC}"; exit 1; }

# Start backend service
uvicorn main:app --host 0.0.0.0 --port 8000
BACKEND_PID=$!
echo -e "${GREEN}Backend service started, PID: $BACKEND_PID${NC}"

# wait for backend service to initialize
echo -e "${BLUE}Waiting for backend service to initialize...${NC}"
sleep 5