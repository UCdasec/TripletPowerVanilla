# Method from "Cross-Device Profiled Side-Channel Attack with Unsupervised Domain Adaptation"

We borrow the code from paper above and customized it to make it running over our dataset.

## Content
```preprocess_data.ipynb``` is to convert our dataset to the format that can work on CDPA method.  

Convert our dataset npz file, which includes attibutes ['power_trace', 'plain_text', 'key'], to multiple npy files. In addtion to that, it also generate corresopinding hamming weight lables. For example,
```PC2_CB2_TDX3_K3_U_20k_0317.npz``` --> ```X_train.npy```, ```X_attack.npy```, ```Y_attack.npy```, ```Y_attack.npy```, ```plaintests_attack.npy```, ```plaintexts_train.npy```, ```Y_attack_hw.npy```, ```Y_train_hw.npy```. 

Where ```PC2_CB2_TDX3_K3_U_20k_0317.npz``` is our collected npz data file, ```X_train.npy``` is the splitted power traces for training; ```X_attack.npy``` is the splitted power trace for attacking; ```Y_train.npy``` is the corresponding labels for training;```Y_attack.npy``` is the corresponding labels for attacking; ```Y_()_hw.npy``` is the corresponding hamming weight labels for training/attacking; ```plaintests_().npy``` is the splitted plain texts for training/attacking.

To run the pre-processing script, you need to configure the parameters, such as the ```target_byte``` (which byte index the attack is to be performed on), ```start_idx``` and ```end_idx``` of attack window. Besides, you also need to sepcify the path of the npz file that need to convert. 

```XMEGA_CDPA_over_our_dataset.ipynb``` is to run CDPA over the converted dataset, which collected by us.

To run CDPA over the converted data from above function, you need to config several parameters:

Markup : * Parameters:
            * ```real_key_01```: the key of the source domain in hex value
            * ```real_key_02```: the key of the target domain in hex value
            * ```source_file_path```: parent folder path for source data
            * ```target_file_path```: parent folder path for target data
            * ```trace_length```: the length of power trace (i.e., length of attack window)
            * you may also need to modify length value based on the size of your data. such as ```train_num```, ```valid_num```, ```target_finetune_num```, ```target_test_num```
            

