import os
import subprocess

sets = ['A', 'B', 'C', 'D', 'E']
exp_name = 'exp_with_number_and_types_new'
numb_iters = 200000
for s in sets:
    
#     # train
#     os.system('python main.py --exp_folder={} --target_set={} --sample_interval={}'.format(exp_name, s, numb_iters))

    # test
    output = subprocess.run('python evaluate_parallel.py --checkpoint=./checkpoints/{}_{}_{}.pth --target_set={}'.format(exp_name, s, numb_iters, s), shell=True, stdout=subprocess.PIPE)
    
    # save results
    text_file = open('./logs/{}_{}_{}.txt'.format(exp_name, numb_iters, s), "w")
    print(output.stdout)
    text_file.write(str(output.stdout))
    text_file.close()
