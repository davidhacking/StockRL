#!/bin/bash
# Launch an experiment using the docker gpu image

cmd_line="$@"

echo "Executing in the docker (gpu image):"
echo $cmd_line

# docker run -it --gpus all -u root --rm --network host --ipc=host \
#   --mount src=$(pwd),target=/home/mamba/stable-baselines3,type=bind stablebaselines/stable-baselines3:latest \
#   bash -c "cd /home/mamba/stable-baselines3/ && $cmd_line"

docker run -it --gpus all -u root --rm --network host --ipc=host \
  --mount src=/home/david/MF,target=/home/mamba/MF,type=bind mystablebaselines_stock_rl \
  bash -c "cd /home/mamba/MF/ && { [ -z \"$cmd_line\" ] && exec bash || exec $cmd_line; }"