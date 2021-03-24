## Label Leakage and Protection Demo with FMNIST Dataset

```FL_Label_Attack_Protection_Demo.py``` shows how we protect label informaiton in the settings of Federated Learning (FL).

### Requirements
* Python 3.x
* Tensorflow 2.x 

We have made some changes to make Marvell be compatible with Tensorflow 1.x.
For example, to make sure that it can be called with ```py_func``` in Tensorflox 1.x, we have
explicitly casted the data type to ```np.float32``` in the code. 
It should work if you copy and paste the corresponding custom gradient functions 
of max_norm and Marvell to your network designed with Tensorflow 1.x.
However, as a cost, it's much slower than the one we are using for the Criteo dataset. We 
strongly recommend to test the code for Criteo dataset. 

### Usage 

* see ```run_gpu.sh```

#### Common Parameters
* --gpu_option: use GPU or not
* --gpu_id: use which gpu
* --num_epochs
* --batch_size
* --num_hints: number of hints for hint attack

#### iso 
* --white_Gaussian_noise: add white Gaussian noise
* --white_Gaussian_stddev: parameter s

Run:

* ```python FMNIST_Label_Attack_Protection_Demo.py --gpu_option --gpu_id 0 --num_epochs 3  --num_hints 1 --white_Gaussian_noise --white_Gaussian_stddev 4.0```



#### max_norm
* --max_norm: use max_norm as protectioin
Run:
* ```python FMNIST_Label_Attack_Protection_Demo.py --gpu_option --gpu_id 1 --num_epochs 3 --num_hints 1 --max_norm```

#### Marvell
*  --sumKL: use Marvell as protection
* --sumKL_threshold: sumKL threshold 

Run:
* ```python FMNIST_Label_Attack_Protection_Demo.py --gpu_option --gpu_id 2 --num_epochs 3 --num_hints 1 --sumKL --sumKL_threshold 0.16```