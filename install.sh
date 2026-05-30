#!/usr/bin/env bash
# ============================================================
#  Mevin - one-command installer for Mac / Linux
#  Installs Python deps, pulls the AI model, starts the app
# ============================================================
set -e
cd "$(dirname "$0")"

echo ""
echo "  Installing Mevin..."
echo ""

# --- Check Python ---
if ! command -v python3 >/dev/null 2>&1; then
  echo "  [!] Python 3 not found."
  echo "      Mac:   brew install python"
  echo "      Linux: sudo apt install python3 python3-pip"
  exit 1
fi

# --- Check Ollama ---
if ! command -v ollama >/dev/null 2>&1; then
  echo "  [!] Ollama not found. Install it:"
  echo "      curl -fsSL https://ollama.com/install.sh | sh"
  echo "      (or download from https://ollama.com)"
  exit 1
fi

echo "  Installing Python packages..."
python3 -m pip install --upgrade pip >/dev/null 2>&1 || true
python3 -m pip install -r requirements.txt

echo ""
read -p "  Enable auto-discovery for IP cameras/NVRs? (y/n): " ONVIF
if [ "$ONVIF" = "y" ] || [ "$ONVIF" = "Y" ]; then
  python3 -m pip install -r requirements-onvif.txt
fi

echo ""
echo "  Pulling the AI model (gemma3:4b, ~3.3GB, one time)..."
ollama pull gemma3:4b

echo ""
echo "  Starting Mevin..."
echo "  Open http://localhost:5555 in your browser."
echo ""
python3 mevin.py
