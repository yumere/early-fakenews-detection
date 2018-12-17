Pytorch implementation for Yang Liu, Yi-Fang Brook Wu ["Early Detection of Fake News on Social Media Through Propagation Path Classification with Recurrent and Convolutional Networks"](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16826). AAAI 2018.  

*Note: We do not publish the dataset used in this code. The authors of this paper also did not publish the dataset*

## Preprocessing
In the project root directory, 

1. Prepare dataset
    ```bash
    python preprocess/prepare_dataset.py --config config.json --max-length 5 --output output_5.json --n_processes 30
    ```
2. split train/dev dataset

    We need to split the dataset into train/dev carefully. 
    If the dataset is split into train/dev randomly, the features used in RNN/CNN may be duplicated. 
    This is because a number of users in the dataset emits several tweets over entire dataset time. 
    ```bash
    python preprocess/split_train_dev.py --input output_5.json -d 2018-01-01 --train_output train_5.json --dev_output dev_5.json
    ```
    
## Train
In the project root directory,

```bash
fakenews_detection/main.py --mode train --config config.json --train_file train_5.json --dev_file dev_5.json --lr 0.2 --batch_size 128 --cuda 1
```