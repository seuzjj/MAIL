# MAIL

**Paper**: Adaptive and Iterative Learning with Multi-perspective Regularizations for Metal Artifact Reduction

**Authors**: Jianjia Zhang, Haiyang Mao, Hengyong Yu, Weiwen Wu* and Dinggang Shen*

Date : August-21-2023  
Version : 1.0

## Requirements and Dependencies
``` bash
pip install -r requirements.txt
```

## Training Demo
``` bash
python MAIL_Train.py --gpu_id 0 --data_path "./test_data" --batchSize 1 --log_dir "logs/" --model_dir "models/" --manualSeed 2000 --T 2
```

## Test Demo
``` bash
python MAIL_Test.py --gpu_id 0 --data_path "./test_data" --model_dir "./models/MAIL_state_64.pt" --T 2
```