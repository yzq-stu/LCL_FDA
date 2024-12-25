## Settings
* The project is tested on Python 3.6, Tensorflow 1.13.1 and cuda 10.0
* Then install the dependencies: ```pip install -r helper_requirements.txt```
* And compile the cuda-based operators: ```sh compile_op.sh```  
(Note: may change the cuda root directory ```CUDA_ROOT``` in ```./util/sampling/compile_ops.sh```)


* Run: ```python utils/data_prepare_[ ].py```  
(Note: may specify other directory as ```dataset_path``` in ```./util/data_prepare_[ ].py```)

## Training/Test
* Training:
```
python -B main.py --gpu 0 --mode train --test_area 3
```  
(Note: specify the `--test_area` from `1~6`)
* Test:
```
python -B tester.py --gpu 0 --mode test --test_area 3 --model_path 'pretrained/Area5/snap-32251'
```  
(Note: specify the `--test_area` index and the trained model path `--model_path`)

The code is built on [RandLA-Net](https://github.com/QingyongHu/RandLA-Net). We thank the authors for sharing the codes.
