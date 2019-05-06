# Deep Landscape Forecasting for Real-time Bidding Advertising
This is the implementation for our KDD 2019 paper "Deep Landscape Forecasting for Real-time Bidding Advertising".

If you have any problems, please feel free to contact the authors [Kan Ren](http://saying.ren), [Jiarui Qin](http://apex.sjtu.edu.cn/members/qinjr) and [Lei Zheng](http://apex.sjtu.edu.cn/members/zhenglei).

### Abstract
> The emergence of real-time auction in online advertising has drawn huge attention of modeling the market competition, i.e., bid landscape forecasting. The problem is formulated as to forecast the probability distribution of market price for each ad auction. With the consideration of the censorship issue which is caused by the second-price auction mechanism, many researchers have devoted their efforts on bid landscape forecasting by incorporating survival analysis from medical research field. However, most existing solutions mainly focus on either counting-based statistics of the segmented sample clusters, or learning a parameterized model based on some heuristic assumptions of distribution forms. Moreover, they neither consider the sequential patterns of the feature over the price space. In order to capture more sophisticated yet flexible patterns at fine-grained level of the data, we propose a Deep Landscape Forecasting (DLF) model which combines deep learning for probability distribution forecasting and survival analysis for censorship handling. Specifically, we utilize a recurrent neural network to flexibly model the conditional winning probability w.r.t. each bid price. Then we conduct the bid landscape forecasting through probability chain rule with strict mathematical derivations. And, in an end-to-end manner, we optimize the model by minimizing two negative likelihood losses with comprehensive motivations. Without any specific assumption for the distribution form of bid landscape, our model shows great advantages over previous works on fitting various sophisticated market price distributions. In the experiments over two large-scale real-world datasets, our model significantly outperforms the state-of-the-art solutions under various metrics.

### Setups


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
