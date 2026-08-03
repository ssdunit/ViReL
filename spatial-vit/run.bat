REM upgrading pip
pip install --upgrade pip
REM installing torch,torchvision
echo "Installing Torch,torchvision"
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
echo "Installing requirements"
uv pip install -r requirements.txt
echo "Required login for hugging face"
hf auth login