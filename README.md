## Label Leakage and Protection in Federated Learning
* For double blind review

## Conda environment 

* ```conda create -n py3_tf2 --file  py3_tf2_package_list.txt```

Or 

* ```conda env create -f py3_tf2.yml```

This downloads the conda packages as a conda environment in your local directories. 
From there, you can activate the environment and start running the code.

* ```conda activate py3_tf2```

## Dataset

* You can test the code by using Criteo and FMNIST datasets. 
* We have sampled 0.5% instances of Criteo dataset for your test.


In the demo of FMNIST, we have made some changes to make Marvell be compatible with Tensorflow 1.x.
For example, to make sure that it can be called with ```py_func``` in Tensorflox 1.x, we have
explicitly casted the data type to ```np.float32``` in the code. 
It should work if you copy and paste the corresponding custom gradient functions 
of max_norm and Marvell to your network designed with Tensorflow 1.x.
However, as a cost, it's much slower than the one we are using for the Criteo dataset. We 
strongly recommend to test the code for Criteo dataset. 


## Run 

* Please go to the corresponding folder and read their own ```README.md``` to run 
corresponding codes. 
* You may need ```chmod 777``` for some scripts.

## Results

In case you have trouble running the code:

* We have put some tensorboard logs of our finished experiments in ```Criteo/logs/```, you can just un-tar them and use ```tensorboard --logdir``` command to view the results. These logs track the history of running the code with the full Criteo dataset.  