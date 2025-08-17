#!/bin/bash
# Cross-platform build script with Docker detection

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}📦 Building A-Share Trading Decision Platform${NC}"

# Function to check if Docker is available
check_docker() {
    if command -v docker &> /dev/null; then
        if docker info &> /dev/null; then
            echo -e "${GREEN}✅ Docker is available and running${NC}"
            return 0
        else
            echo -e "${YELLOW}⚠️  Docker is installed but not running${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}⚠️  Docker is not installed${NC}"
        return 1
    fi
}

# Function to build with Docker
build_with_docker() {
    echo -e "${GREEN}🐳 Building with Docker...${NC}"
    
    # Create Dockerfile if it doesn't exist
    if [ ! -f "Dockerfile" ]; then
        cat > Dockerfile << 'EOF'
# Multi-stage build for A-Share Trading Platform
FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim AS backend-builder
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY .cache/ ./.cache/

FROM python:3.11-slim AS production
WORKDIR /app
RUN apt-get update && apt-get install -y redis-server && rm -rf /var/lib/apt/lists/*
COPY --from=backend-builder /app ./
COPY --from=frontend-builder /app/frontend/build ./frontend/build
EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF
        echo -e "${GREEN}✅ Created Dockerfile${NC}"
    fi
    
    # Create docker-compose.yml
    cat > docker-compose.yml << 'EOF'
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./.cache:/app/.cache
      - ./reports:/app/reports
    environment:
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
  
volumes:
  redis_data:
EOF
    
    # Build Docker image
    docker build -t ashore-trading-platform .
    
    echo -e "${GREEN}✅ Docker build completed${NC}"
    echo -e "${BLUE}To run with Docker:${NC}"
    echo -e "  docker-compose up -d"
    echo -e "  open http://localhost:8000"
}

# Function to build natively
build_native() {
    echo -e "${GREEN}🖥️  Building natively...${NC}"
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo -e "${YELLOW}⚠️  Virtual environment not found. Running setup...${NC}"
        ./setup.sh
    fi
    
    # Activate virtual environment
    echo -e "${GREEN}🔧 Activating virtual environment...${NC}"
    source venv/bin/activate
    
    # Install build dependencies
    echo -e "${GREEN}📋 Installing build dependencies...${NC}"
    pip install pyinstaller==6.2.0
    
    # Build frontend if it exists
    if [ -d "frontend" ]; then
        echo -e "${GREEN}🌐 Building frontend...${NC}"
        cd frontend
        if [ -f "package.json" ]; then
            npm install
            npm run build
        fi
        cd ..
    fi
    
    # Clean previous builds
    echo -e "${YELLOW}🧹 Cleaning previous builds...${NC}"
    rm -rf build/ dist/ *.spec
    
    # Create standalone executable
    echo -e "${GREEN}🔨 Creating standalone server...${NC}"
    pyinstaller --onefile \
        --name "ashore-trading-server" \
        --add-data "src:src" \
        --add-data ".cache:.cache" \
        --add-data "reports:reports" \
        --hidden-import uvicorn \
        --hidden-import fastapi \
        --hidden-import akshare \
        --hidden-import lightgbm \
        --hidden-import polars \
        src/main.py
    
    # Create distribution package
    echo -e "${BLUE}📦 Creating distribution package...${NC}"
    mkdir -p dist/ashore-trading-platform
    cp dist/ashore-trading-server dist/ashore-trading-platform/
    cp -r frontend/build dist/ashore-trading-platform/frontend/ 2>/dev/null || echo "Frontend not found, skipping..."
    cp README.md dist/ashore-trading-platform/
    cp holdings.json.example dist/ashore-trading-platform/
    
    # Create startup script
    cat > dist/ashore-trading-platform/start.sh << 'EOF'
#!/bin/bash
echo "Starting A-Share Trading Platform..."
echo "Server will be available at: http://localhost:8000"
./ashore-trading-server --production
EOF
    chmod +x dist/ashore-trading-platform/start.sh
    
    # Create ZIP for distribution
    cd dist
    zip -r "A-Share-Trading-Platform.zip" ashore-trading-platform/
    cd ..
    
    echo -e "${GREEN}✅ Native build completed${NC}"
    echo -e "${BLUE}To run:${NC}"
    echo -e "  cd dist/ashore-trading-platform"
    echo -e "  ./start.sh"
}

# Main build logic
if check_docker; then
    echo -e "${BLUE}Choose build method:${NC}"
    echo -e "  1. Docker (recommended for consistency)"
    echo -e "  2. Native (faster startup)"
    read -p "Enter choice (1 or 2): " choice
    
    case $choice in
        1)
            build_with_docker
            ;;
        2)
            build_native
            ;;
        *)
            echo -e "${YELLOW}Invalid choice, using Docker by default${NC}"
            build_with_docker
            ;;
    esac
else
    echo -e "${BLUE}Docker not available, building natively...${NC}"
    build_native
fi

echo -e "${GREEN}🎉 Build completed successfully!${NC}"