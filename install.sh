#!/bin/bash
set -e

echo "=== wavetoearth installer ==="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check Python
echo "Checking Python..."
if command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PY_MAJOR=$(echo $PY_VERSION | cut -d. -f1)
    PY_MINOR=$(echo $PY_VERSION | cut -d. -f2)
    if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 10 ]; then
        echo -e "${GREEN}Python $PY_VERSION found${NC}"
    else
        echo -e "${RED}Python 3.10+ required, found $PY_VERSION${NC}"
        exit 1
    fi
else
    echo -e "${RED}Python3 not found. Please install Python 3.10+${NC}"
    exit 1
fi

# Check/Install Rust
echo ""
echo "Checking Rust..."
if command -v cargo &> /dev/null; then
    RUST_VERSION=$(rustc --version | cut -d' ' -f2)
    echo -e "${GREEN}Rust $RUST_VERSION found${NC}"
else
    echo -e "${YELLOW}Rust not found. Installing...${NC}"
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    echo -e "${GREEN}Rust installed${NC}"
fi

# Clone or update repo
echo ""
INSTALL_DIR="${WAVETOEARTH_DIR:-$HOME/wavetoearth}"

if [ -d "$INSTALL_DIR" ]; then
    echo "Updating existing installation at $INSTALL_DIR..."
    cd "$INSTALL_DIR"
    git pull
else
    echo "Cloning wavetoearth to $INSTALL_DIR..."
    git clone https://github.com/2opk/wavetoearth.git "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip3 install --upgrade pip
pip3 install maturin

# Build Rust extension
echo ""
echo "Building Rust extension (this may take a few minutes)..."
cd wavetoearth_core
maturin develop --release
cd ..

# Install Python package
echo ""
echo "Installing wavetoearth..."
pip3 install -e .

# Verify installation
echo ""
echo "Verifying installation..."
if python3 -c "import wavetoearth_core; print('Rust extension OK')" 2>/dev/null; then
    echo -e "${GREEN}Rust extension loaded successfully${NC}"
else
    echo -e "${RED}Failed to load Rust extension${NC}"
    exit 1
fi

# Setup Claude Code MCP
echo ""
echo "=== Claude Code MCP Setup ==="

MCP_CMD="$INSTALL_DIR/mcp_server.py"

if command -v claude &> /dev/null; then
    echo "Adding wavetoearth MCP server to Claude Code..."
    claude mcp add wavetoearth --scope user -- python3 "$MCP_CMD"
    echo -e "${GREEN}MCP server added${NC}"
else
    echo -e "${YELLOW}Claude Code CLI not found.${NC}"
    echo ""
    echo "To manually configure, add this to ~/.claude.json:"
    echo ""
    echo '{'
    echo '  "mcpServers": {'
    echo '    "wavetoearth": {'
    echo "      \"command\": \"python3\","
    echo "      \"args\": [\"$MCP_CMD\"]"
    echo '    }'
    echo '  }'
    echo '}'
fi

echo ""
echo "=== Installation complete! ==="
echo ""
echo "Usage:"
echo "  1. Open Claude Code"
echo "  2. Type /mcp to verify wavetoearth is loaded"
echo "  3. Ask Claude to load a VCD/FST file:"
echo "     'Load /path/to/simulation.vcd and find where it hangs'"
echo ""
