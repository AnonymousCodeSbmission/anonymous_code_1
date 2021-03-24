python FMNIST_Label_Attack_Protection_Demo.py --gpu_option --gpu_id 0 --num_epochs 3  --num_hints 1 --white_Gaussian_noise --white_Gaussian_stddev 4.0
python FMNIST_Label_Attack_Protection_Demo.py --gpu_option --gpu_id 1 --num_epochs 3 --num_hints 1 --max_norm
python FMNIST_Label_Attack_Protection_Demo.py --gpu_option --gpu_id 2 --num_epochs 3 --num_hints 1 --sumKL --sumKL_threshold 0.16
