# E1-246-Assignment1

## About
This repository contains the implementation of the word2vec model (skip-gram architecture) as proposed by Mikolov et al. in the paper *[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)* .

## How to run

### Dataset
To train the model we use the Reuters corpus available through the nltk api. `helper.py` contains the required methods to read and process the required data. 

### Setting the hyperparameters
All the hyperparameters are loaded from `config.json` file. Here I have briefly described the meaning of each of these hyperparameter flags present in `config.json`.
* `learning_rate` : learning rate required to train the model
* `batch_size` : training batch size
* `epochs` : number of epochs to train
* `win_radius` : number of words to the side of a center word (specify as 2 if you want the total window size as 5)
* `neg_sampes` : number of negative samples to be considered while calculating the loss
* `embed_dims` : embedding dimensions
* `subsample` : if true then some of the most frequent words will be dropped with some probability (see paper)
* `remove_stopwords` : if true then stopwords will be removed (**Note**: For this keep subsample true as well, if remove_stopwords is set to true then probabilistic subsampling is not performed)
* `subsample_threshold` : set the subsample threshold (default is 1e-5 as mentioned in the paper)
* `low_freq_threshold` : words less than `low_freq_threshold` will be removed
* `simlex_threshold` : while calculating simlex scores consider only the word pairs >= simlex_threshold (0 to consider all words)
* `final_loss_batch_size` : while calculating the final loss on the trained model on train/val/test sets consider this as a batch size and report the average loss over total number of batches. (**Note**: This is different from `batch_size` to speed up model evaluation as while training `batch_size` can be 1)
* `train_loss_disp_freq` : frequency of display for train loss
* `val_loss_disp_freq` : frequency of display for val loss
* `k_most_common` : this is required for the analogy task, where `k_most_common` words are considered (see [1])

### How to run

To train your model run:

```
python word2vec.py --mode train --save_path ckpt/model --config_file ckpt/model_b1ns200ws5ed60ep5nsw/config.json
```

