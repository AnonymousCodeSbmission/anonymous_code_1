import os
import subprocess
import time


FNULL = open(os.devnull, 'w')
common_options = [
    'python3',
    '../main_wdl.py',
    '--gpu_option',
    '--num_epochs', '10',
    '--batch_size', '10240', #'32768',
    '--lr_schedule', '1e-4',
    '--train_file', '../data/new_0.005train_0.9.csv',
    '--test_file', '../data/new_0.005test_0.1.csv',
    '--model_config', '128', '128', '128', '-1', '128', '128', '128',
    '--period', '100',
    '--num_hints', '1', '3', '5',
]

gpus = {}
for gpu_id in range(8):
    gpus[gpu_id] = ['--device_number', str(gpu_id),]


##### KL
# init_scale, gpu_index
KL_configs = [
    # (0.01, 0),
    # (0.05, 0),
    # (0.1, 0),
    # (0.15, 1),
    # (0.2, 1),
    # (0.25, 1)
    # (0.3, 2),
    # (0.4, 2),
    # (0.5, 2),
    (1.0, 3),
    # (1.5, 3),
    # (1.75, 3),
    # (2.0, 4),
    # (2.5, 4),
    # (3.0, 4),
    # (4.0, 5),
    # (5.0, 5),
    # (6.0, 5),
    # (7.0, 6),
    # (8.0, 6),
    # (9.0, 6),
    # (10.0, 7),
    # (20.0, 7),
    # (25.0, 7)
]

for init_scale, gpu_idx in KL_configs:
    KL_specific_options = [
        '--noise_layer_function', 'sumKL',
        '--p_frac', 'pos_frac',
        '--init_scale', str(init_scale),
        '--uv_choice', 'uv',
    ]
    subprocess.Popen(args=common_options + gpus[gpu_idx] + KL_specific_options, stdout=FNULL)

# ratio, gpu_idx
white_gaussian_configs = [
    # (0, 0),
    # (0.5, 0),
    # (1.0, 0),
    # (1.25, 1),
    # (1.5, 1),
    # (1.75, 1),
    # (2, 2),
    (2.5, 2),
    # (3, 2),
    # (3.5, 3),
    # (4.0, 3),
    # (4.5, 3),
    # (5, 4),
    # (7, 4),
    # (9, 4),
    # (11, 5),
    # (0.25, 5),
    # (15.0, 5),
    # (25.0, 6),
    # (0.75, 6),
    # (2.25, 6),
    # (2.75, 7),
    # (6.0, 7)
]

for ratio, gpu_idx in white_gaussian_configs:
    white_gaussian_options = [
        '--noise_layer_function', 'white_gaussian',
        '--ratio', str(ratio)]
    subprocess.Popen(args=common_options + gpus[gpu_idx] + white_gaussian_options, stdout=FNULL)

#### expectation_align
expectation_specific_options = [
    '--noise_layer_function', 'expectation',
]
subprocess.Popen(args=common_options + gpus[7] + expectation_specific_options, stdout=FNULL)

# time.sleep(2) # sleep 1 second




FNULL.close()
