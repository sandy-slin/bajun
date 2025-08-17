#!/bin/bash
# Development server startup script

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting A-Share Trading Decision Platform Development Server${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found. Running setup...${NC}"
    ./setup.sh
fi

# Activate virtual environment
echo -e "${GREEN}📦 Activating virtual environment...${NC}"
source venv/bin/activate

# Check if Redis is running (required for caching)
if ! pgrep -x "redis-server" > /dev/null; then
    echo -e "${YELLOW}⚠️  Redis server not running. Starting Redis...${NC}"
    if command -v redis-server &> /dev/null; then
        redis-server --daemonize yes --port 6379
        echo -e "${GREEN}✅ Redis server started on port 6379${NC}"
    else
        echo -e "${RED}❌ Redis not installed. Please install Redis: brew install redis${NC}"
        exit 1
    fi
fi

# Check if required environment variables are set
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠️  .env file not found. Creating default configuration...${NC}"
    cat > .env << EOF
# Development Configuration
ENVIRONMENT=development
DEBUG=true

# API Configuration
DEEPSEEK_API_KEY=sk-f4affcb7b78243f5a138e7c9bdbbd6ee

# Database Configuration
REDIS_URL=redis://localhost:6379/0
SQLITE_DB_PATH=.cache/trading_data.db

# Cache Configuration
CACHE_TTL_MINUTES=60
DATA_REFRESH_INTERVAL=300

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=.cache/app.log

# Performance Configuration
MAX_WORKERS=4
REQUEST_TIMEOUT=30
EOF
    echo -e "${GREEN}✅ Created .env file with default configuration${NC}"
fi

# Install/update dependencies
echo -e "${GREEN}📋 Checking dependencies...${NC}"
pip install -r requirements.txt --quiet

# Create necessary directories
mkdir -p .cache/{data,models,logs}
mkdir -p reports/{daily,portfolio,behavior}

# Start FastAPI development server with hot reload
echo -e "${GREEN}🌐 Starting FastAPI development server...${NC}"
echo -e "${GREEN}🔗 Server will be available at: http://localhost:8000${NC}"
echo -e "${GREEN}📚 API documentation: http://localhost:8000/docs${NC}"
echo -e "${GREEN}🛠️  Alternative UI: http://localhost:8501 (Streamlit)${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop the server${NC}"
echo ""

# Start the development server with hot reload
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload --log-level info