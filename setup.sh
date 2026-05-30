#!/usr/bin/env bash
# CouncilAI Environment Setup Script
# Generates a secure .env file with a high-entropy JWT secret.

set -euo pipefail

ENV_FILE=".env"
EXAMPLE_FILE=".env.example"

# Colors for modern terminal logs
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}===================================================${NC}"
echo -e "${BLUE}        CouncilAI Environment Setup                ${NC}"
echo -e "${BLUE}===================================================${NC}"

# Read existing env values if .env already exists
GEMINI_KEY=""
OPENROUTER_KEY=""
NVIDIA_KEY=""
HF_TOKEN_VAL=""
MOCK_LLM_VAL="false"
JWT_SECRET_VAL=""

if [ -f "$ENV_FILE" ]; then
    echo -e "${YELLOW}Existing .env file detected.${NC}"
    # Read existing values using grep/sed to avoid shell injection
    GEMINI_KEY=$(grep -E "^GEMINI_API_KEY=" "$ENV_FILE" | cut -d'=' -f2- | tr -d '"' | tr -d "'") || true
    OPENROUTER_KEY=$(grep -E "^OPENROUTER_API_KEY=" "$ENV_FILE" | cut -d'=' -f2- | tr -d '"' | tr -d "'") || true
    NVIDIA_KEY=$(grep -E "^NVIDIA_NIM_API_KEY=" "$ENV_FILE" | cut -d'=' -f2- | tr -d '"' | tr -d "'") || true
    HF_TOKEN_VAL=$(grep -E "^HF_TOKEN=" "$ENV_FILE" | cut -d'=' -f2- | tr -d '"' | tr -d "'") || true
    MOCK_LLM_VAL=$(grep -E "^MOCK_LLM=" "$ENV_FILE" | cut -d'=' -f2- | tr -d '"' | tr -d "'") || true
    JWT_SECRET_VAL=$(grep -E "^JWT_SECRET=" "$ENV_FILE" | cut -d'=' -f2- | tr -d '"' | tr -d "'") || true
fi

# Generate a high-entropy cryptographically secure random JWT secret if not already set
if [ -z "$JWT_SECRET_VAL" ] || [ "$JWT_SECRET_VAL" = "change-this-to-a-secure-random-string" ]; then
    echo -e "Generating a secure, high-entropy JWT secret..."
    if command -v openssl >/dev/null 2>&1; then
        JWT_SECRET_VAL=$(openssl rand -hex 32)
    else
        JWT_SECRET_VAL=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1)
    fi
fi

# Ask for credentials interactively (press Enter to keep existing/default)
echo -e "\nPlease enter your API keys (press Enter to keep existing/blank):"

read -p "Gemini API Key [${GEMINI_KEY:-None}]: " temp_gemini
GEMINI_KEY=${temp_gemini:-$GEMINI_KEY}

read -p "OpenRouter API Key [${OPENROUTER_KEY:-None}]: " temp_or
OPENROUTER_KEY=${temp_or:-$OPENROUTER_KEY}

read -p "NVIDIA NIM API Key [${NVIDIA_KEY:-None}]: " temp_nv
NVIDIA_KEY=${temp_nv:-$NVIDIA_KEY}

read -p "HuggingFace Token (for gated local models) [${HF_TOKEN_VAL:-None}]: " temp_hf
HF_TOKEN_VAL=${temp_hf:-$HF_TOKEN_VAL}

# Generate the .env file
cat << EOF > "$ENV_FILE"
# CouncilAI Environment Secrets Configuration
# -------------------------------------------------------------
# Sensitive API keys and credentials.
# Note: For general non-sensitive settings, use config.yaml.

# ── API Keys ──────────────────────────────────────────────────
GEMINI_API_KEY="${GEMINI_KEY}"
OPENROUTER_API_KEY="${OPENROUTER_KEY}"
NVIDIA_NIM_API_KEY="${NVIDIA_KEY}"

# ── HuggingFace Token ─────────────────────────────────────────
HF_TOKEN="${HF_TOKEN_VAL}"

# ── JWT Security Secret ───────────────────────────────────────
JWT_SECRET="${JWT_SECRET_VAL}"

# ── Testing / Development ─────────────────────────────────────
MOCK_LLM="${MOCK_LLM_VAL:-false}"
EOF

echo -e "\n${GREEN}Successfully created/updated .env file with secure secrets!${NC}"
echo -e "${GREEN}JWT Secret is securely configured.${NC}"
echo -e "To launch the full suite, run:"
echo -e "  ${BLUE}docker compose up --build${NC}"
echo -e "To launch local models alongside the services, run:"
echo -e "  ${BLUE}docker compose --profile local-models up --build${NC}"
echo -e "${BLUE}===================================================${NC}"
