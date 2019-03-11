# E1-246-Assignment1

## About
This repository contains the implementation of the word2vec model (skip-gram architecture) as proposed by Mikolov et al. in the paper *[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)* .

## Setup Instructions

### Dataset
To train the model I have used the Reuters corpus available through the nltk api. `helper.py` contains the required methods to read and process the required data. 

### Setting the hyperparameters
All the hyperparameters are loaded from `config.json` file. Here I have briefly described each of these hyperparameter flags present in `config.json`.
* `learning_rate` : learning rate required to train the model
* `batch_size` : training batch size
* `epochs` : number of epochs to train
* `win_radius` : number of words to the side of a center word (specify as 2 if you want the total window size as 5)
* `neg_sampes` : number of negative samples to be considered while calculating the loss
* `embed_dims` : embedding dimensions
* `subsample` : if true then some of the most frequent words will be dropped with some probability (see paper)
* `remove_stopwords` : if true then stopwords will be removed (**Note**: For this keep subsample true as well, if `remove_stopwords` is set to true then probabilistic subsampling is not performed)
* `subsample_threshold` : set the subsample threshold (default is 1e-5 as mentioned in the paper)
* `low_freq_threshold` : words less than `low_freq_threshold` will be removed
* `simlex_threshold` : while calculating simlex scores consider only the word pairs >= `simlex_threshold` (set it to 0 to consider all words)
* `final_loss_batch_size` : while calculating the final loss on the trained model on train/val/test sets consider this as a batch size and report the average loss over total number of batches. (**Note**: A separate batch size is used for final loss calculation in order to speed up the loss calculation as training `batch_size` can be 1)
* `train_loss_disp_freq` : frequency of display for train loss
* `val_loss_disp_freq` : frequency of display for val loss
* `k_most_common` : this is required for the analogy task, where `k_most_common` words are considered (see [1])
* `analogy_top_k`: if target word occurs in 0:analogy_top_k predictions then it is counted towards calculation of accuracy
* `print_top_k`: number of predictions to print in test and analogy tasks
### How to run

All the below commands assume that `config.json` is present in the same directory as the code. If you wish to load `config.json` from some other directory then please specify that using `--config_file` flag as `--config_file dirname/config.json` in all of the commands below

To train your model run:
```
python word2vec.py --mode train --save_path ckpt/model_60d_ep3/ 
```
To test your model on specified words run:
```
python word2vec.py --mode test --test_words company shares one --load_path ckpt/model_60d_ep3/ 
```
**Note**: If `--test_words` flag is not used then the program will use the test set to generate the results.

To evaluate the model on Simlex-999 run:
```
python word2vec.py --mode simlex --data_path eval_data/ --load_path ckpt/model_60d_ep3/
```
To evaluate the model on analogy task run:
```
python word2vec.py --mode analogy --data_path eval_data/ --load_path ckpt/model_60d_ep3/
```
To visualize the words run:
```
python word2vec.py --mode visualize --load_path ckpt/model_60d_ep3/
```
## Pretrained Model
The `ckpt` directory contains the trained models for 60-d and 100-d embeddings trained on 3 epochs and 5 epochs which can be used to reproduce some of the results in the report.

## References
<cite>[1] Mikolov, Tomas, et al. "Efficient Estimation of Word Representations in Vector Space." 2013.</cite> <br>
<cite>[2] Mikolov, Tomas, et al. "Distributed Representations of Words and Phrases and their Compositionality." 2013.</cite>
