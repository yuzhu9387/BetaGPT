#!/bin/bash

# show colored output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

echo -e "${BLUE}===== BetaGPT start service =====${NC}"

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
echo -e "${BLUE}Installing backend dependencies...${NC}"
cd backend || { echo -e "${RED}Failed to enter backend directory${NC}"; exit 1; }
pip3 install --no-user -r requirements.txt || { echo -e "${RED}Failed to install dependencies${NC}"; exit 1; }

# Start backend service
echo -e "${GREEN}Starting backend service...${NC}"
python3 main.py &
BACKEND_PID=$!
echo -e "${GREEN}Backend service started, PID: $BACKEND_PID${NC}"

# wait for backend service to initialize
echo -e "${BLUE}Waiting for backend service to initialize...${NC}"
sleep 5

# Start frontend service
echo -e "${BLUE}Starting frontend service...${NC}"
cd ../frontend || { echo -e "${RED}Failed to enter frontend directory${NC}"; exit 1; }

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo -e "${YELLOW}npm not found, adding Node.js module to Replit...${NC}"
    
    # Add Node.js module to .replit file if it doesn't exist
    if ! grep -q "nodejs" .replit; then
        echo -e "${BLUE}Adding Node.js module to .replit file...${NC}"
        cd ../
        if grep -q "modules" .replit; then
            # If modules line exists, add nodejs to it
            sed -i 's/modules = \[/modules = \["nodejs",/g' .replit
        else
            # If modules line doesn't exist, add it
            echo 'modules = ["nodejs", "python-3.12", "web"]' >> .replit
        fi
        cd frontend
    fi
    
    echo -e "${YELLOW}Node.js module added. You may need to restart your Repl for changes to take effect.${NC}"
    echo -e "${RED}Please restart the Repl and run the script again.${NC}"
    kill $BACKEND_PID
    exit 1
fi

# Install frontend dependencies
echo -e "${BLUE}Installing frontend dependencies...${NC}"
npm install || { 
    echo -e "${RED}Failed to install frontend dependencies${NC}"
    kill $BACKEND_PID
    exit 1
}

# Start frontend service
echo -e "${GREEN}Starting frontend service...${NC}"
npm start &
FRONTEND_PID=$!
echo -e "${GREEN}Frontend service started, PID: $FRONTEND_PID${NC}"

echo -e "${GREEN}All services started!${NC}"
echo -e "${BLUE}Press Ctrl+C to stop all services${NC}"

# catch interrupt signal, gracefully stop services
trap 'echo -e "${BLUE}stopping services...${NC}"; kill $BACKEND_PID $FRONTEND_PID; echo -e "${GREEN}services stopped${NC}"; exit 0' INT

# keep script running
wait 