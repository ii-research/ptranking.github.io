<img src=https://github.com/microsoft/LightGBM/blob/master/docs/logo/LightGBM_logo_black_text.svg width=300 />

# LightGBM Source Code Analysis

[LightGBM](https://github.com/microsoft/LightGBM/)

## Preparing the development environment
macOS Catalina 10.15.6 CPU version

```
brew install cmake
brew install libomp
git clone --recursive https://github.com/ii-research/ptranking.github.io.git ; cd ptranking.github.io/tutorial
mkdir build ; cd build
cmake ..
make -j4
```
## Testing examples provided by the source code
[LightGBM/examples/lambdarank](https://github.com/microsoft/LightGBM/tree/master/examples/lambdarank)

cd to the examples/lambdarank directory and execute

```
# Training
"../../lightgbm" config=train.conf
# Prediction
"../../lightgbm" config=predict.conf
```
Training results:

```
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
...

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
In the train.conf file:objective = lambdarank.
If you want to customize the loss function, you need to change this.
## Core loss function
[LightGBM/src/objective/rank_objective.hpp](https://github.com/microsoft/LightGBM/blob/master/src/objective/rank_objective.hppf)

The task is focused on analyzing the source code here.

* RankingObjective Class: class RankingObjective : public ObjectiveFunction {}
    - public methods
        - void Init(const Metadata& metadata, data_size_t num_data)
            - num_data_
            - label_
            - weights_
            - query_boundaries_
            - num_queries_
            
            No need to implement it yourself.
        - void GetGradients(
                    const double* score, 
                    score_t* gradients,
                    score_t* hessians)

            No need to implement it yourself.
        - virtual void GetGradientsForOneQuery(
                        data_size_t query_id, 
                        data_size_t cnt,
                        const label_t* label,
                        const double* score, score_t* lambdas,
                        score_t* hessians) const = 0;
        
            Define a virtual function, you need to implement it yourself.

            The key to customizing the loss function is to implement the GetGradientsForOneQuery method.
        - const char* GetName() const override { return "gyrank"; }

            It needs to be implemented on its own.
        - std::string ToString() const override {} 

            It needs to be implemented on its own.
        - bool NeedAccuratePrediction() const override { return false; }

            It needs to be implemented on its own.
    - protected data members
        - int seed_;
        - data_size_t num_queries_;  /*! \brief Number of data */ 
        - data_size_t num_data_;  
        - const label_t* label_;  /*! \brief Pointer of label */
        - const label_t* weights_; /*! \brief Pointer of weights */
        - const data_size_t* query_boundaries_; /*! \brief Query boundries */

* LambdarankNDCG Class: class LambdarankNDCG : public RankingObjective {}
    - public methods
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
    - private data members
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
## First noob loss function

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
## LambdaRank--GetGradientsForOneQuery相关变量详解
NDCG=DCG/MaxDCG

<!-- |  LightGBM   | PTranking  | xx|
|  ----  | ----  |----|
| inverse_max_dcg  | batch_std_diffs | # standard pairwise differences, i.e., S_{ij} |
| 单元格  | batch_pred_s_ij | # computing pairwise differences, i.e., s_i - s_j |
| 单元格  | batch_delta_ndcg |
| 单元格  | batch_loss = torch.sum((batch_loss_1st + batch_loss_2nd) * batch_delta_ndcg * 0.5)  |
| 单元格  | pair_row_inds |
| 单元格  | pair_col_inds | -->
>  Function Parameter
|  LightGBM | Type |Description |
|  ----  | ----  |  ----  |
| query_id | data_size_t(int32) | each query group's max dcg |
| cnt | double | current query group's max dcg |
| label| label_t* (float*)| get sorted indices for scores |
| score | double* | each query group's max dcg |
| lambdas | score_t* (float*) | current query group's max dcg |
| hessians| score_t* (float*) | get sorted indices for scores |



> Initializing Variables

|  LightGBM | Type |Description |
|  ----  | ----  |  ----  |
| inverse_max_dcg_ | array | each query group's max dcg |
| inverse_max_dcg | double | current query group's max dcg |
| sorted_idx[]| int array | get sorted indices for scores |

> Get best and worst score

||||
|  ----  | ----  |  ----  |  ----  |
| best_score | double | = score[sorted_idx[0]] |get best score |
| worst_idx |int| = cnt - 1 | current group last index |
| worst_score | double | = score[sorted_idx[worst_idx]] |get worst score|
|sum_lambdas|double|=0|initializing sum_lambdas|

> start accmulate lambdas by pairs
```
  for (data_size_t i = 0; i < cnt; ++i) {
```
|high|data_size_t(int32)|=sorted_idx[i]|
|high_score|double|=score[high]|?|

||||
|  ----  | ----  |  ----  |
| high_label| int | static_cast<int>(label[high]) |
| high_label_gain| double | =label_gain_[high_label]; |
| high_discount| double | =DCGCalculator::GetDiscount(i) |
| high_sum_lambda| double | = 0.0 |
| high_sum_hessian| double | = 0.0 |
```
    for (data_size_t j = 0; j < cnt; ++j) {
```
||||
|  ----  | ----  |  ----  |
| low| data_size_t(int32) | = sorted_idx[j]|
| low_label| int | = static_cast<int>(label[low]) |
|low_score|double|= score[low]|

> only consider pair with different label

||||
| delta_score | double | = high_score - low_score |
| low_label_gain|  double | = label_gain_[low_label] |
| low_discount|  double | = DCGCalculator::GetDiscount(j) |

||||
|  ----  | ----  |  ----  |
| dcg_gap  | double | = high_label_gain - low_label_gain |get dcg gap |
| paired_discount | double | = fabs(high_discount - low_discount) | get discount of this pair |
| delta_pair_NDCG | double | = dcg_gap * paired_discount * inverse_max_dcg |get delta NDCG |

> regular the delta_pair_NDCG by score distance

```
        if (norm_ && best_score != worst_score) {
          delta_pair_NDCG /= (0.01f + fabs(delta_score));
        }
```

> calculate lambda for this pair

||||
|  ----  | ----  |  ----  |
|  p_lambda  | double | = GetSigmoid(delta_score) |calculate lambda for this pair  |
|  p_hessian  | double| = p_lambda * (1.0f - p_lambda)|calculate lambda for this pair  |
|  sum_lambdas  | double | -= 2 * p_lambda |lambda is negative, so use minus to accumulate  |

> update lambda

```
        // calculate lambda for this pair
        double p_lambda = GetSigmoid(delta_score);
        double p_hessian = p_lambda * (1.0f - p_lambda);
        // update
        p_lambda *= -sigmoid_ * delta_pair_NDCG;
        p_hessian *= sigmoid_ * sigmoid_ * delta_pair_NDCG;
        high_sum_lambda += p_lambda;
        high_sum_hessian += p_hessian;
        lambdas[low] -= static_cast<score_t>(p_lambda);
        hessians[low] += static_cast<score_t>(p_hessian);
        // lambda is negative, so use minus to accumulate
        sum_lambdas -= 2 * p_lambda;

```
||||
|  ----  | ----  |  ----  |
|lambdas[low]|x|x|
|hessians[low]|x|x|
| norm_factor | double |std::log2(1 + sum_lambdas) / sum_lambdas|
|  lambdas[]  | x  |
|  hessians[]  | x  |

> after for loop (j) update
```
      // update
      lambdas[high] += static_cast<score_t>(high_sum_lambda);
      hessians[high] += static_cast<score_t>(high_sum_hessian);
```
||||
|  ----  | ----  |  ----  |
|lambdas[high]|score_t(float)|x|
|hessians[high]|score_t(float)|x|

> after for loop (i) 
||||
|  ----  | ----  |  ----  |
| norm_factor | double |std::log2(1 + sum_lambdas) / sum_lambdas|
|  lambdas[]  | x  |
|  hessians[]  | x  |

```
    if (norm_ && sum_lambdas > 0) {
      double norm_factor = std::log2(1 + sum_lambdas) / sum_lambdas;
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = static_cast<score_t>(lambdas[i] * norm_factor);
        hessians[i] = static_cast<score_t>(hessians[i] * norm_factor);
      }
    }
```



LightGBM-LambdaRank
```
- metadata
    - num_data_: Number of training data rows.(rank.train)
```



PTranking-LambdaRank
```
#- gradient -#
batch_grad = sigma * (0.5 * (1 - batch_std_Sij) - reciprocal_1_add_exp_sigma(batch_pred_s_ij, sigma=sigma))
batch_grad = batch_grad * batch_delta_ndcg
batch_grad = torch.sum(batch_grad, dim=1, keepdim=True) # relying on the symmetric property, i-th row-sum corresponding to the cumulative gradient w.r.t. i-th document.
ctx.save_for_backward(batch_grad)
```