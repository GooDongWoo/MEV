# Multi-Exit Vision Transformer (MEViT)

## Project Structure

### Core Model Files
- `mevit_model.py`: Main architecture definition of Multi-Exit Vision Transformer
- `mevit_integrate_ee.py`: Implementation of Early Exit integration functionality  
- `mevit_ddp_train.py`: Distributed training implementation using DistributedDataParallel

### Training & Testing
- `mevit_train.py`: Main training script for MEViT model
- `mevit_test.py`: Model testing and FLOPS calculation
- `backbone_train.py`: Backbone model training
- `backbone_test.py`: Backbone model testing
- `test.sh`: Shell script for test execution

### Data Loading & Preprocessing
- `Dloaders.py`: Data loader implementation (optimized numworkers)
- `imagenet_load.py`: ImageNet dataset loading functionality

### Cache & Result Files
- `cache_exp_cifar10.pt`: Experiment cache for CIFAR-10
- `cache_exp_cifar100.pt`: Experiment cache for CIFAR-100 
- `cache_exp_imagenet.pt`: Experiment cache for ImageNet
- `cache_result_mevit_cifar10.pt`: Results cache for CIFAR-10
- `cache_result_mevit_cifar100.pt`: Results cache for CIFAR-100
- `make_last_cache.py`: Final cache generation

### Experiments & Analysis
- `experiments_cifar10.ipynb`: CIFAR-10 experiment notebook
- `experiments_cifar100.ipynb`: CIFAR-100 experiment notebook
- `experiments_imagenet.ipynb`: ImageNet experiment notebook
- `mevit_spec_save.ipynb`: Model specification saving and visualization
- `my_plot.ipynb`: Results visualization
- `tmp.ipynb`: Temporary experiment notebook

### Calibration & Scaling
- `temperature_scaling.py`: Temperature scaling implementation
- `maxtrix_scaling.py`: Matrix scaling functionality

### Configuration & Misc
- `.vscode`: Visual Studio Code settings
- `.gitignore`: Git ignore configuration  
- `README.md`: Project documentation
- `runs`: Execution logs directory

## Recent Updates
- Added ImageNet dataset support
- Optimized data loader with increased number of workers
- Implemented FLOPS calculation
- Added matrix scaling functionality
- Enhanced experiment process documentation

## Datasets
The project includes experiments on three datasets:
- CIFAR-10
- CIFAR-100  
- ImageNet-1K

## Note
This repository implements Multi-Exit Vision Transformer and conducts experiments across multiple datasets. Recent updates focus on ImageNet integration, dataloader optimization, and FLOPS calculation capabilities.