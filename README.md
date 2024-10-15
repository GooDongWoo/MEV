# Introduction

This code was developed and executed using Jupyter notebooks.

The following instructions assume Ubuntu 20.04 operating system with superuser access, Nvidia GPUs, GPU drivers already installed and CUDA version 10.1, 11.0 or 11.2.

# Setting Up the Environment

1. [Install Docker](https://docs.docker.com/engine/install/ubuntu/)
2. [Install Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit)
3. `sudo docker run --gpus all -it -p 8888:8888 tensorflow/tensorflow:2.3.2-gpu-jupyter` (also tested with the `tensorflow/tensorflow:2.4.1-gpu-jupyter` image)
4. Copy the URL provided in docker logs (including the token).
5. <kbd>CTRL</kbd>+<kbd>P</kbd> then <kbd>CTRL</kbd>+<kbd>Q</kbd> to detach from the container without terminating the execution.
6. Install SciPy inside the container: `sudo docker exec -it [container_name] bash` (you can find the container name from the output of `sudo docker ps`) then `pip install scipy==1.5` (use <kbd>CTRL</kbd>+<kbd>D</kbd> to terminate and detach)
7. Paste the copied URL in your browser to open Jupyter (if you are running the docker container on a remote server, you need to replace the IP address with that of the server).
8. Upload all of the `.ipynb` files in this repository.

# Running the Experiments

Note: in each of the notebooks, you can modify `SELECTED_GPUS` to specify which GPUs to use. If you only have a single GPU available, set `SELECTED_GPUS = [0]`. The distributed training may not be supported in some notebooks.

1. Run the `train_cifar10_backbone`, `train_cifar100_backbone`, `train_fashion_mnist_backbone` and`train_disco_backbone` notebooks to train the backbones.

2. Run the `precompute_cifar_features`, `precompute_disco_features` and `precompute_fashion_mnist_features` notebooks to precompute the intermediate representations of the backbones.

3. Run the `ee` and `cw` notebooks to run the end-to-end and classifier-wise experiments, respectively. You can change the `dataset`, `head_type`, `version` and other parameters given to the `train` function.

4. Run the `calculate_flops` notebook to calculate the FLOPS, the `calculate_maes` notebook to calculate MAEs for the DISCO dataset cases, and the `plots` notebook to draw the plots.


