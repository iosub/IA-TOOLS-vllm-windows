"c:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat"
set DISTUTILS_USE_SDK=1
set VLLM_TARGET_DEVICE=cuda
#(replace 10 with your desired cpu threads to use in parallel to speed up compilation)
set MAX_JOBS=10

pip install torch==2.7.1+cu126 torchaudio==2.7.1+cu126 torchvision==0.22.1+cu126 --index-url https://download.pytorch.org/whl/cu126
