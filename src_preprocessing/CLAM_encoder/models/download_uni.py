import os
from huggingface_hub import login, hf_hub_download

login()  # Authenticate Hugging Face

model_name = 'uni_v2' # uni_v1 or uni_v2
if model_name == 'uni_v1':
    download_dir = "MahmoodLab/UNI"
elif model_name == 'uni_v2':
    download_dir = "MahmoodLab/UNI2-h"


local_dir = "/home/espis/espis/CLAM_modified_UNI_20250203/models/"
os.makedirs(local_dir, exist_ok=True)  # Ensure the directory exists

# Download model without creating a symlink
model_path = hf_hub_download(
    download_dir, 
    filename=f"pytorch_model.bin", 
    local_dir=local_dir, 
    force_download=True,
    cache_dir=local_dir  # Ensures the actual file is stored here
)

os.rename(local_dir+"pytorch_model.bin", local_dir+f"{model_name}.bin")
print(f"Model downloaded to: {model_path}")
