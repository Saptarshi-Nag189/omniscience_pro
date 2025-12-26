#!/bin/bash
# Omniscience Pro - Docker Quick Start Script
# Auto-detects GPU availability

set -e

echo "==================================="
echo "  Omniscience Pro - Docker Setup"
echo "==================================="

# Check prerequisites
check_prereqs() {
    echo "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker not found. Install: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    echo "✅ Docker found"
}

# Detect GPU
detect_gpu() {
    if docker info 2>/dev/null | grep -q "Runtimes.*nvidia"; then
        echo "✅ NVIDIA GPU detected - using GPU acceleration"
        COMPOSE_FILE="docker-compose.yml"
    else
        echo "ℹ️  No NVIDIA GPU detected - using CPU mode"
        COMPOSE_FILE="docker-compose.cpu.yml"
    fi
}

# Build and start services
start_services() {
    echo ""
    echo "Using: $COMPOSE_FILE"
    echo "Building and starting services..."
    
    sudo docker compose -f "$COMPOSE_FILE" up -d --build
    
    echo ""
    echo "Waiting for Ollama to be ready..."
    sleep 15
    
    echo ""
    echo "==================================="
    echo "  ✅ Omniscience Pro is Ready!"
    echo "==================================="
    echo ""
    echo "  Web UI:  http://localhost:8501"
    echo "  Ollama:  http://localhost:11434"
    echo ""
    echo "  Commands:"
    echo "    Stop:   sudo docker compose -f $COMPOSE_FILE down"
    echo "    Logs:   sudo docker compose -f $COMPOSE_FILE logs -f"
    echo "    Status: sudo docker compose -f $COMPOSE_FILE ps"
    echo ""
}

# Main
check_prereqs
detect_gpu
start_services
