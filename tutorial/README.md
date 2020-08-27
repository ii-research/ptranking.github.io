<img src=https://github.com/microsoft/LightGBM/blob/master/docs/logo/LightGBM_logo_black_text.svg width=300 />

# LightGBM Source Code Analysis

[LightGBM](https://github.com/microsoft/LightGBM/)

## Preparing the development environment
macOS Catalina 10.15.6 CPU version
```
brew install cmake
brew install libomp
git clone --recursive https://github.com/ii-research/ptranking.github.io.git ; cd ptranking.github.io
mkdir build ; cd build
cmake ..
make -j4
sudo make install
```
## 测试源码提供的例子
[LightGBM/examples/lambdarank](https://github.com/microsoft/LightGBM/tree/master/examples/lambdarank)

cd到examples/lambdarank目录下，执行
```
# Training
"../../lightgbm" config=train.conf
# Prediction
"../../lightgbm" config=predict.conf
```
Training的结果:
```
# kou2n @ MacBook-Pro-2 in ~/Projects/LightGBM/examples/lambdarank on git:master x [17:23:58] 
$ "../../lightgbm" config=train.conf  
[LightGBM] [Info] Finished loading parameters
[LightGBM] [Info] Loading query boundaries...
[LightGBM] [Info] Loading query boundaries...
[LightGBM] [Info] Finished loading data in 0.118956 seconds
[LightGBM] [Warning] Auto-choosing col-wise multi-threading, the overhead of testing was 0.008773 seconds.
You can set `force_col_wise=true` to remove the overhead.
[LightGBM] [Info] Total Bins 6179
[LightGBM] [Info] Number of data points in the train set: 3005, number of used features: 211
[LightGBM] [Info] Finished initializing training
[LightGBM] [Info] Started training...
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[LightGBM] [Info] Iteration:1, training ndcg@1 : 0.713528
[LightGBM] [Info] Iteration:1, training ndcg@3 : 0.698799
[LightGBM] [Info] Iteration:1, training ndcg@5 : 0.720909
[LightGBM] [Info] Iteration:1, valid_1 ndcg@1 : 0.449524
[LightGBM] [Info] Iteration:1, valid_1 ndcg@3 : 0.511721
[LightGBM] [Info] Iteration:1, valid_1 ndcg@5 : 0.565374
[LightGBM] [Info] 0.003292 seconds elapsed, finished iteration 1
.....

[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[LightGBM] [Info] Iteration:99, training ndcg@1 : 0.994504
[LightGBM] [Info] Iteration:99, training ndcg@3 : 0.990198
[LightGBM] [Info] Iteration:99, training ndcg@5 : 0.983637
[LightGBM] [Info] Iteration:99, valid_1 ndcg@1 : 0.610286
[LightGBM] [Info] Iteration:99, valid_1 ndcg@3 : 0.644799
[LightGBM] [Info] Iteration:99, valid_1 ndcg@5 : 0.679468
[LightGBM] [Info] 0.675337 seconds elapsed, finished iteration 99
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[LightGBM] [Info] Iteration:100, training ndcg@1 : 0.997347
[LightGBM] [Info] Iteration:100, training ndcg@3 : 0.990904
[LightGBM] [Info] Iteration:100, training ndcg@5 : 0.984666
[LightGBM] [Info] Iteration:100, valid_1 ndcg@1 : 0.607429
[LightGBM] [Info] Iteration:100, valid_1 ndcg@3 : 0.642375
[LightGBM] [Info] Iteration:100, valid_1 ndcg@5 : 0.675462
[LightGBM] [Info] 0.679780 seconds elapsed, finished iteration 100
[LightGBM] [Info] Finished training
```
train.conf文件中objective = lambdarank。如果自定义损失函数，需要修改此处。
## 核心损失函数
[LightGBM/src/objective/rank_objective.hpp](https://github.com/microsoft/LightGBM/blob/master/src/objective/rank_objective.hppf)

任务重点是分析这里的源码。
* RankingObjective类: class RankingObjective : public ObjectiveFunction {}
    - public方法
        - void Init(const Metadata& metadata, data_size_t num_data)
            - num_data_
            - label_
            - weights_
            - query_boundaries_
            - num_queries_
            
            不需要自己实现。
        - void GetGradients(
                    const double* score, 
                    score_t* gradients,
                    score_t* hessians)

            不需要自己实现。
        - virtual void GetGradientsForOneQuery(
                        data_size_t query_id, 
                        data_size_t cnt,
                        const label_t* label,
                        const double* score, score_t* lambdas,
                        score_t* hessians) const = 0;
        
            定义虚函数，具体方法需要自己实现。

            自定义函数关键是实现GetGradientsForOneQuery方法。
        - const char* GetName() const override { return "gyrank"; }

            需要自己实现。
        - std::string ToString() const override {} 

            暂时不用管。
        - bool NeedAccuratePrediction() const override { return false; }

            暂时不用管。
    - protected 数据成员
        - int seed_;
        - data_size_t num_queries_;  /*! \brief Number of data */ 
        - data_size_t num_data_;  /*! \brief Pointer of label */
        - const label_t* label_;  /*! \brief Pointer of weights */
        - const label_t* weights_;  /*! \brief Query boundries */
        - const data_size_t* query_boundaries_;

* LambdarankNDCG类: class LambdarankNDCG : public RankingObjective {}
    - public方法
        - void Init(const Metadata& metadata, data_size_t num_data)
            - RankingObjective::Init(metadata, num_data);
            - DCGCalculator::CheckLabel(label_, num_data_);
            - inverse_max_dcgs_.resize(num_queries_);
        - inline void GetGradientsForOneQuery(
                            data_size_t query_id, 
                            data_size_t cnt,
                            const label_t* label, 
                            const double* score,
                            score_t* lambdas,
                            score_t* hessians
                            )
        - inline double GetSigmoid(double score)
        - void ConstructSigmoidTable()

            construct sigmoid table to speed up sigmoid transform

        - const char* GetName() const override { return "lambdarank"; }
    - private 数据成员
        - double sigmoid_;  /*! \brief Simgoid param */
        - bool norm_;  /*! \brief Normalize the lambdas or not */
        - int truncation_level_;   /*! \brief Truncation position for max DCG */
        - std::vector<double> inverse_max_dcgs_;  /*! \brief Cache inverse max DCG, speed up calculation */
        - std::vector<double> sigmoid_table_;  /*! \brief Cache result for sigmoid transform to speed up */
        - std::vector<double> label_gain_;  /*! \brief Gains for labels */
        - size_t _sigmoid_bins = 1024 * 1024;  /*! \brief Number of bins in simoid table */
        - double min_sigmoid_input_ = -50;  /*! \brief Minimal input of sigmoid table */
        - double max_sigmoid_input_ = 50;  /*! \brief Maximal input of sigmoid table */
        - double sigmoid_table_idx_factor_;  /*! \brief Factor that covert score to bin in sigmoid table */
## 第一个测试损失函数
```
$ lightgbm config=/Users/kou2n/Projects/LightGBM/examples/gyndcg/train.conf
[LightGBM] [Info] Finished loading parameters
[LightGBM] [Info] Loading query boundaries...
[LightGBM] [Info] Loading query boundaries...
[LightGBM] [Info] Finished loading data in 0.139901 seconds
[LightGBM] [Warning] Auto-choosing row-wise multi-threading, the overhead of testing was 0.004316 seconds.
You can set `force_row_wise=true` to remove the overhead.
And if memory is not enough, you can set `force_col_wise=true`.
[LightGBM] [Info] Total Bins 6179
[LightGBM] [Info] Number of data points in the train set: 3005, number of used features: 211
[LightGBM] [Info] Finished initializing training
[LightGBM] [Info] Started training...
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[LightGBM] [Info] Iteration:1, training ndcg@1 : 0.305236
[LightGBM] [Info] Iteration:1, training ndcg@3 : 0.392209
[LightGBM] [Info] Iteration:1, training ndcg@5 : 0.450073
[LightGBM] [Info] Iteration:1, valid_1 ndcg@1 : 0.327429
[LightGBM] [Info] Iteration:1, valid_1 ndcg@3 : 0.357957
[LightGBM] [Info] Iteration:1, valid_1 ndcg@5 : 0.411907
[LightGBM] [Info] 0.002575 seconds elapsed, finished iteration 1
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[LightGBM] [Info] Iteration:2, training ndcg@1 : 0.31258
[LightGBM] [Info] Iteration:2, training ndcg@3 : 0.397724
[LightGBM] [Info] Iteration:2, training ndcg@5 : 0.463167
[LightGBM] [Info] Iteration:2, valid_1 ndcg@1 : 0.270857
[LightGBM] [Info] Iteration:2, valid_1 ndcg@3 : 0.333107
[LightGBM] [Info] Iteration:2, valid_1 ndcg@5 : 0.391865
[LightGBM] [Info] 0.005594 seconds elapsed, finished iteration 2
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[LightGBM] [Info] Iteration:3, training ndcg@1 : 0.309548
[LightGBM] [Info] Iteration:3, training ndcg@3 : 0.392534
[LightGBM] [Info] Iteration:3, training ndcg@5 : 0.462557
[LightGBM] [Info] Iteration:3, valid_1 ndcg@1 : 0.290667
[LightGBM] [Info] Iteration:3, valid_1 ndcg@3 : 0.311596
[LightGBM] [Info] Iteration:3, valid_1 ndcg@5 : 0.387377
...

[LightGBM] [Info] Iteration:99, training ndcg@1 : 0.312817
[LightGBM] [Info] Iteration:99, training ndcg@3 : 0.400553
[LightGBM] [Info] Iteration:99, training ndcg@5 : 0.45607
[LightGBM] [Info] Iteration:99, valid_1 ndcg@1 : 0.301333
[LightGBM] [Info] Iteration:99, valid_1 ndcg@3 : 0.358342
[LightGBM] [Info] Iteration:99, valid_1 ndcg@5 : 0.420757
[LightGBM] [Info] 0.247832 seconds elapsed, finished iteration 99
[LightGBM] [Warning] No further splits with positive gain, best gain: -inf
[LightGBM] [Info] Iteration:100, training ndcg@1 : 0.318977
[LightGBM] [Info] Iteration:100, training ndcg@3 : 0.403002
[LightGBM] [Info] Iteration:100, training ndcg@5 : 0.457984
[LightGBM] [Info] Iteration:100, valid_1 ndcg@1 : 0.301333
[LightGBM] [Info] Iteration:100, valid_1 ndcg@3 : 0.36297
[LightGBM] [Info] Iteration:100, valid_1 ndcg@5 : 0.419104
[LightGBM] [Info] 0.250646 seconds elapsed, finished iteration 100
[LightGBM] [Info] Finished training
```
