#upgrading pip
pip install --upgrade pip
#installing torch,torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install -r requirements.txt
echo "Required login for hugging face"
hf auth login