https://pytorch-extension.intel.com/installation?platform=gpu&version=v2.7.10%2Bxpu&os=linux%2Fwsl2&package=pip

sudo apt-get install intel-opencl-icd

python3 -m venv .venv
source .venv/bin/activate

python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/xpu
python -m pip install intel-extension-for-pytorch==2.7.10+xpu oneccl_bind_pt==2.7.0+xpu --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/

pip install timm==1.0.15 scikit-learn==1.7.0 matplotlib==3.10.3

python -c "import torch; import intel_extension_for_pytorch as ipex; print(torch.__version__); print(ipex.__version__); [print(f'[{i}]: {torch.xpu.get_device_properties(i)}') for i in range(torch.xpu.device_count())];"

