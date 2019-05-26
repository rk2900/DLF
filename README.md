## Deep Landscape Forecasting for Real-time Bidding Advertising
This is the implementation for our KDD 2019 paper "[Deep Landscape Forecasting for Real-time Bidding Advertising](https://arxiv.org/abs/1905.03028)".

The preprint version of the paper has been published on Arxiv: https://arxiv.org/abs/1905.03028.

If you have any problems, please feel free to contact the authors [Kan Ren](http://saying.ren), [Jiarui Qin](http://apex.sjtu.edu.cn/members/qinjr) and [Lei Zheng](mailto:zhenglei2016@sjtu.edu.cn).

### Abstract
> The emergence of real-time auction in online advertising has drawn huge attention of modeling the market competition, i.e., bid landscape forecasting. The problem is formulated as to forecast the probability distribution of market price for each ad auction. With the consideration of the censorship issue which is caused by the second-price auction mechanism, many researchers have devoted their efforts on bid landscape forecasting by incorporating survival analysis from medical research field. However, most existing solutions mainly focus on either counting-based statistics of the segmented sample clusters, or learning a parameterized model based on some heuristic assumptions of distribution forms. Moreover, they neither consider the sequential patterns of the feature over the price space. In order to capture more sophisticated yet flexible patterns at fine-grained level of the data, we propose a Deep Landscape Forecasting (DLF) model which combines deep learning for probability distribution forecasting and survival analysis for censorship handling. Specifically, we utilize a recurrent neural network to flexibly model the conditional winning probability w.r.t. each bid price. Then we conduct the bid landscape forecasting through probability chain rule with strict mathematical derivations. And, in an end-to-end manner, we optimize the model by minimizing two negative likelihood losses with comprehensive motivations. Without any specific assumption for the distribution form of bid landscape, our model shows great advantages over previous works on fitting various sophisticated market price distributions. In the experiments over two large-scale real-world datasets, our model significantly outperforms the state-of-the-art solutions under various metrics.

### Setups
We recommend the settings of Tensorflow (>=1.3) and Python (2.7.6).

The models are trained under the same hardware settings with an Intel(R) Core(TM) i7-6900K CPU processor, an NVIDIA GeForce GTX 1080Ti GPU processor and 128 GB memory. The training time of each compared model is less than ten hours (as reported from the slowest training model MTLSA) on each dataset.

All the models are trained until convergence and we consider learning rate from {1e-4, 1e-5, 1e-3, 1e-3}. The value of $\alpha$ is tuned to 0.25. Batch size is fixed on 128 and embedding dimension is 32. All the deep learning models take input features and feed through an embedding layer for the subsequent feedforward calculation. The hyperparameters of each model are tuned and the best performances have been reported.

### Data Preparation
The full dataset can be downloaded at [this link](http://bit.ly/2VTMMdm) and the corresponding MD5 code is `841698b0dd8718b1b4a4ff2e54bb72b4`.

The raw data of iPinYou can be downloaded from [Dropbox](https://www.dropbox.com/s/txz0ms0axqf7jrl/ipinyou.contest.dataset.7z?dl=0).

The feature engineering code is [here](https://github.com/rk2900/make-ipinyou-data), which is forked and slightly different to the original repository.

### Data specification
Each subset of the data contains `.yzbx.txt`, `featureindex.txt` and `.log.txt`.
We created the first data file `.log.txt` from the raw data of the original data source (please refer to our paper).
Then we made feature engineering according to the created feature dictionary `featindex.txt`.
The corresponding feature engineered data are in `.yzbx.txt`.

If you need to reproduce the experiemtns, you may run over `.yzbx.txt`.

In `yzbx.txt` file, each line is a sample containing the "`yzbx`" data, the information is splitted by `SPACE`.
Here `z` is the true market price, `b` is the proposed bid price and `x` is the list of features (multi-hot encoded as `feat_id:1`).
In the experiment, we only use `zbx` data.
Note that, for the uncensored data, `z < b`, while for the censored data, `z >= b`.

### Run the Codes
The running command are listed as below.
```
python km.py             # Kaplan-Meier
python gamma_model.py    # Gamma
python cox.py            # Lasso-Cox and DeepSurv
python deephit.py        # DeepHit
python DWPP.py           # DWPP
python RNN.py 0.0001      # for RNN
python DLF.py 0.0001     # for DLF
```

### Citations
```
@inproceedings{ren2019deep,
  title={Deep Landscape Forecasting for Real-time Bidding Advertising},
  author={Ren, Kan and Qin, Jiarui and Zheng, Lei and Zhang, Weinan and Yu, Yong},
  booktitle={Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  year={2019},
  organization={ACM}
}
```
