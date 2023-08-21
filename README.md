# MAIL
Code for paper： Adaptive and Iterative Learning with Multi-perspective Regularizations for Metal Artifact Reduction

#Training

python MAIL_Train.py --gpu_id 0 --data_path "./test_data" --batchSize 1 --log_dir "logs/" --model_dir "models/" --manualSeed 2000 --T 2 --resume 64

#Test

python MAIL_Test.py --gpu_id 0 --data_path "./test_data" --model_dir "/data/Mhy/Code/Ablation/MAIL/models/MAIL_state_64.pt" --T 2
