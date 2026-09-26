@echo off
REM Build the shared conda environment for all four baselines (YOLOv8n, YOLO11n, RemDet-Tiny, DEIM-N).
REM Run from the repository root in an Anaconda Prompt:   env\setup_env.bat
REM Needs an NVIDIA driver that supports CUDA 12.1 (nvidia-smi: "CUDA Version" >= 12.1).

set ENV_NAME=litegtr-baselines

call conda create -n %ENV_NAME% -y python=3.10 || goto :fail
call conda activate %ENV_NAME% || goto :fail
python -m pip install -U pip || goto :fail

echo.
echo [1/3] torch 2.1.2 + torchvision 0.16.2 (CUDA 12.1)
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121 || goto :fail

echo.
echo [2/3] mmcv 2.1.0 (prebuilt package only, never compiled locally)
pip install mmcv==2.1.0 --only-binary mmcv -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html || goto :mmcv_fail

echo.
echo [3/3] everything else (env\requirements.txt)
pip install -r env\requirements.txt || goto :fail

echo.
python -c "import torch, torchvision, mmcv; from mmcv.ops import nms; print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| mmcv', mmcv.__version__, '| CUDA available:', torch.cuda.is_available())" || goto :fail
python train_baselines.py --dry-run
echo.
echo Done. Next time: conda activate %ENV_NAME%  then  python train_baselines.py
exit /b 0

:mmcv_fail
echo.
echo mmcv 2.1.0 has no prebuilt package for this Python/CUDA. Open
echo   https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
echo search for win_amd64, and change python=3.10 at the top of this file to a listed cp3xx version.
echo Then: conda env remove -n %ENV_NAME%   and run this script again. See README.md.
exit /b 1

:fail
echo.
echo Setup failed at the step above. See README.md.
exit /b 1
