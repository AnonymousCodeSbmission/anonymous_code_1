## Label Leakage and Protection Demo with Criteo Dataset

### Requirements

* Python 3.x
* Tensorflow 2.x
* DeepCTR (https://github.com/shenweichen/DeepCTR)


### Usage 

* cd to  ```scripts/``` folder
* You may have to use ```chmod 777``` to run the scripts
* We have sampled 0.005 of Criteo dataset for demonstration and put it in the ```data/``` folder
* run corresponding scripts for different protection methods

#### iso
* ```./run_gpu_lable_protection_white_gaussian.sh```

#### max_norm
* ```./run_gpu_lable_protection_max_norm.sh```

#### Marvell
* ```./run_gpu_lable_protection_marvell.sh```

### Run all configurations of different protection methods
* ```python ./run_script_all.py```
