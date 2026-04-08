export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
CUDA_VISIBLE_DEVICES=1 python -m src.eval_mars --experiment_name="osnet_market"