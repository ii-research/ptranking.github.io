/*!
 * Copyright (c) 2020 Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE file in the project root for
 * license information.
 */
#ifndef LIGHTGBM_OBJECTIVE_RANK_OBJECTIVE_HPP_
#define LIGHTGBM_OBJECTIVE_RANK_OBJECTIVE_HPP_

#include <LightGBM/metric.h>
#include <LightGBM/objective_function.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace LightGBM {

/*!
 * \brief Objective function for Ranking
 */
class RankingObjective : public ObjectiveFunction {
 public:
  explicit RankingObjective(const Config& config)
      : seed_(config.objective_seed) {}

  explicit RankingObjective(const std::vector<std::string>&) : seed_(0) {}

  ~RankingObjective() {}
  
  // @brief: 初始化RankingObjective基类
  // @param: metadata,num_data(32位int类型)
  // @ret: 
  // @birth: created by gy on 20200824
  void Init(const Metadata& metadata, data_size_t num_data) override {
    num_data_ = num_data;
    // get label
    label_ = metadata.label();
    // get weights
    weights_ = metadata.weights();
    // get boundries
    query_boundaries_ = metadata.query_boundaries();
    if (query_boundaries_ == nullptr) {
      Log::Fatal("Ranking tasks require query information");
    }
    num_queries_ = metadata.num_queries();
  }
  // @brief: 获得梯度
  // @param: score,gradients,hessians
  // @ret: 
  // @birth: created by gy on 20200824
  void GetGradients(const double* score, score_t* gradients,
                    score_t* hessians) const override {
#pragma omp parallel for schedule(guided)
    for (data_size_t i = 0; i < num_queries_; ++i) {
      const data_size_t start = query_boundaries_[i];
      const data_size_t cnt = query_boundaries_[i + 1] - query_boundaries_[i];
      GetGradientsForOneQuery(i, cnt, label_ + start, score + start,
                              gradients + start, hessians + start);
      if (weights_ != nullptr) {
        for (data_size_t j = 0; j < cnt; ++j) {
          gradients[start + j] =
              static_cast<score_t>(gradients[start + j] * weights_[start + j]);
          hessians[start + j] =
              static_cast<score_t>(hessians[start + j] * weights_[start + j]);
        }
      }
    }
  }

  // @brief: 获得一条query的梯度
  // @param: query_id,cnt,   label,score,lambdas,hessians
  // @ret: 
  // @birth: created by gy on 20200824
  virtual void GetGradientsForOneQuery(data_size_t query_id, data_size_t cnt,
                                       const label_t* label,
                                       const double* score, score_t* lambdas,
                                       score_t* hessians) const = 0;

  const char* GetName() const override = 0;

  std::string ToString() const override {
    std::stringstream str_buf;
    str_buf << GetName();
    return str_buf.str();
  }

  bool NeedAccuratePrediction() const override { return false; }

 protected:
  int seed_;
  data_size_t num_queries_;
  /*! \brief Number of data */
  data_size_t num_data_;
  /*! \brief Pointer of label */
  const label_t* label_;
  /*! \brief Pointer of weights */
  const label_t* weights_;
  /*! \brief Query boundries */
  const data_size_t* query_boundaries_;
};

/*!
 * \brief Objective function for Lambdrank with NDCG
 */
class LambdarankNDCG : public RankingObjective {
 public:
  explicit LambdarankNDCG(const Config& config)
      : RankingObjective(config),
        sigmoid_(config.sigmoid),
        norm_(config.lambdarank_norm),
        truncation_level_(config.lambdarank_truncation_level) {
    label_gain_ = config.label_gain;
    // initialize DCG calculator
    DCGCalculator::DefaultLabelGain(&label_gain_);
    DCGCalculator::Init(label_gain_);
    sigmoid_table_.clear();
    inverse_max_dcgs_.clear();
    if (sigmoid_ <= 0.0) {
      Log::Fatal("Sigmoid param %f should be greater than zero", sigmoid_);
    }
  }

  explicit LambdarankNDCG(const std::vector<std::string>& strs)
      : RankingObjective(strs) {}

  ~LambdarankNDCG() {}
  // @brief: 初始化数据，inverse_max_dcgs_,每个query组的max dcg
  // @param: metadata,num_data(32位int类型)
  // @ret: 
  // @birth: created by gy on 20200826
  void Init(const Metadata& metadata, data_size_t num_data) override {
    RankingObjective::Init(metadata, num_data);

    // @brief: Check the label range for NDCG and lambdarank
    // @param: label Pointer of label,num_data Number of data
    // @ret: 
    // @birth: created by gy on 20200909
    DCGCalculator::CheckLabel(label_, num_data_);
    /*! \brief Cache inverse max DCG, speed up calculation */
    inverse_max_dcgs_.resize(num_queries_);
#pragma omp parallel for schedule(static)
    for (data_size_t i = 0; i < num_queries_; ++i) {
      //every group max_dcgs
      inverse_max_dcgs_[i] = DCGCalculator::CalMaxDCGAtK(
          truncation_level_, label_ + query_boundaries_[i],
          query_boundaries_[i + 1] - query_boundaries_[i]);

      if (inverse_max_dcgs_[i] > 0.0) {
        inverse_max_dcgs_[i] = 1.0f / inverse_max_dcgs_[i];
      }
    }
    // construct sigmoid table to speed up sigmoid transform
    ConstructSigmoidTable();
  }

  inline void GetGradientsForOneQuery(data_size_t query_id, data_size_t cnt,
                                      const label_t* label, const double* score,
                                      score_t* lambdas,
                                      score_t* hessians) const override {
    // get max DCG on current query
    const double inverse_max_dcg = inverse_max_dcgs_[query_id];
    // initialize with zero
    for (data_size_t i = 0; i < cnt; ++i) {
      lambdas[i] = 0.0f;
      hessians[i] = 0.0f;
    }
    // get sorted indices for scores
    std::vector<data_size_t> sorted_idx(cnt);
    for (data_size_t i = 0; i < cnt; ++i) {
      sorted_idx[i] = i;
    }
    // @brief: 得分快速排序，遇到两个数相等时，不对其交换顺序
    // @param: 
    // @ret: 
    // @birth: created by gy on 20200909
    std::stable_sort(
        sorted_idx.begin(), sorted_idx.end(),
        [score](data_size_t a, data_size_t b) { return score[a] > score[b]; });
    // get best and worst score
    const double best_score = score[sorted_idx[0]];
    data_size_t worst_idx = cnt - 1;
    if (worst_idx > 0 && score[sorted_idx[worst_idx]] == kMinScore) {
      worst_idx -= 1;
    }
    const double worst_score = score[sorted_idx[worst_idx]];
    double sum_lambdas = 0.0;
    // start accmulate lambdas by pairs
    for (data_size_t i = 0; i < cnt; ++i) {
      const data_size_t high = sorted_idx[i];
      const int high_label = static_cast<int>(label[high]);
      const double high_score = score[high];
      // high_score为负无穷大时跳过本次循环
      if (high_score == kMinScore) {
        continue;
      }
      const double high_label_gain = label_gain_[high_label];
      const double high_discount = DCGCalculator::GetDiscount(i);
      double high_sum_lambda = 0.0;
      double high_sum_hessian = 0.0;
      for (data_size_t j = 0; j < cnt; ++j) {
        // skip same data
        if (i == j) {
          continue;
        }
        const data_size_t low = sorted_idx[j];
        const int low_label = static_cast<int>(label[low]);
        const double low_score = score[low];
        // only consider pair with different label
        if (high_label <= low_label || low_score == kMinScore) {
          continue;
        }

        const double delta_score = high_score - low_score;

        const double low_label_gain = label_gain_[low_label];
        const double low_discount = DCGCalculator::GetDiscount(j);
        // get dcg gap
        const double dcg_gap = high_label_gain - low_label_gain;
        // get discount of this pair
        const double paired_discount = fabs(high_discount - low_discount);
        // get delta NDCG
        double delta_pair_NDCG = dcg_gap * paired_discount * inverse_max_dcg;
        // regular the delta_pair_NDCG by score distance
        if (norm_ && best_score != worst_score) {
          delta_pair_NDCG /= (0.01f + fabs(delta_score));
        }
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
      }
      // update
      lambdas[high] += static_cast<score_t>(high_sum_lambda);
      hessians[high] += static_cast<score_t>(high_sum_hessian);
    }
    if (norm_ && sum_lambdas > 0) {
      double norm_factor = std::log2(1 + sum_lambdas) / sum_lambdas;
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = static_cast<score_t>(lambdas[i] * norm_factor);
        hessians[i] = static_cast<score_t>(hessians[i] * norm_factor);
      }
    }
  }

  inline double GetSigmoid(double score) const {
    if (score <= min_sigmoid_input_) {
      // too small, use lower bound
      return sigmoid_table_[0];
    } else if (score >= max_sigmoid_input_) {
      // too large, use upper bound
      return sigmoid_table_[_sigmoid_bins - 1];
    } else {
      return sigmoid_table_[static_cast<size_t>((score - min_sigmoid_input_) *
                                                sigmoid_table_idx_factor_)];
    }
  }

  void ConstructSigmoidTable() {
    // get boundary
    min_sigmoid_input_ = min_sigmoid_input_ / sigmoid_ / 2;
    max_sigmoid_input_ = -min_sigmoid_input_;
    sigmoid_table_.resize(_sigmoid_bins);
    // get score to bin factor
    sigmoid_table_idx_factor_ =
        _sigmoid_bins / (max_sigmoid_input_ - min_sigmoid_input_);
    // cache
    for (size_t i = 0; i < _sigmoid_bins; ++i) {
      const double score = i / sigmoid_table_idx_factor_ + min_sigmoid_input_;
      sigmoid_table_[i] = 1.0f / (1.0f + std::exp(score * sigmoid_));
    }
  }

  const char* GetName() const override { return "lambdarank"; }

 private:
  /*! \brief Simgoid param */
  double sigmoid_;
  /*! \brief Normalize the lambdas or not */
  bool norm_;
  /*! \brief Truncation position for max DCG */
  int truncation_level_;
  /*! \brief Cache inverse max DCG, speed up calculation */
  std::vector<double> inverse_max_dcgs_;
  /*! \brief Cache result for sigmoid transform to speed up */
  std::vector<double> sigmoid_table_;
  /*! \brief Gains for labels */
  std::vector<double> label_gain_;
  /*! \brief Number of bins in simoid table */
  size_t _sigmoid_bins = 1024 * 1024;
  /*! \brief Minimal input of sigmoid table */
  double min_sigmoid_input_ = -50;
  /*! \brief Maximal input of sigmoid table */
  double max_sigmoid_input_ = 50;
  /*! \brief Factor that covert score to bin in sigmoid table */
  double sigmoid_table_idx_factor_;
};

/*!
 * \brief Implementation of the learning-to-rank objective function, XE_NDCG
 * [arxiv.org/abs/1911.09798].
 */
class RankXENDCG : public RankingObjective {
 public:
  explicit RankXENDCG(const Config& config) : RankingObjective(config) {}

  explicit RankXENDCG(const std::vector<std::string>& strs)
      : RankingObjective(strs) {}

  ~RankXENDCG() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    RankingObjective::Init(metadata, num_data);
    for (data_size_t i = 0; i < num_queries_; ++i) {
      rands_.emplace_back(seed_ + i);
    }
  }

  inline void GetGradientsForOneQuery(data_size_t query_id, data_size_t cnt,
                                      const label_t* label, const double* score,
                                      score_t* lambdas,
                                      score_t* hessians) const override {
    // Skip groups with too few items.
    if (cnt <= 1) {
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = 0.0f;
        hessians[i] = 0.0f;
      }
      return;
    }

    // Turn scores into a probability distribution using Softmax.
    std::vector<double> rho(cnt, 0.0);
    Common::Softmax(score, rho.data(), cnt);

    // An auxiliary buffer of parameters used to form the ground-truth
    // distribution and compute the loss.
    std::vector<double> params(cnt);

    double inv_denominator = 0;
    for (data_size_t i = 0; i < cnt; ++i) {
      params[i] = Phi(label[i], rands_[query_id].NextFloat());
      inv_denominator += params[i];
    }
    // sum_labels will always be positive number
    inv_denominator = 1. / std::max<double>(kEpsilon, inv_denominator);

    // Approximate gradients and inverse Hessian.
    // First order terms.
    double sum_l1 = 0.0;
    for (data_size_t i = 0; i < cnt; ++i) {
      double term = -params[i] * inv_denominator + rho[i];
      lambdas[i] = static_cast<score_t>(term);
      // Params will now store terms needed to compute second-order terms.
      params[i] = term / (1. - rho[i]);
      sum_l1 += params[i];
    }
    // Second order terms.
    double sum_l2 = 0.0;
    for (data_size_t i = 0; i < cnt; ++i) {
      double term = rho[i] * (sum_l1 - params[i]);
      lambdas[i] += static_cast<score_t>(term);
      // Params will now store terms needed to compute third-order terms.
      params[i] = term / (1. - rho[i]);
      sum_l2 += params[i];
    }
    for (data_size_t i = 0; i < cnt; ++i) {
      lambdas[i] += static_cast<score_t>(rho[i] * (sum_l2 - params[i]));
      hessians[i] = static_cast<score_t>(rho[i] * (1.0 - rho[i]));
    }
  }

  double Phi(const label_t l, double g) const {
    return Common::Pow(2, static_cast<int>(l)) - g;
  }

  const char* GetName() const override { return "rank_xendcg"; }

 private:
  mutable std::vector<Random> rands_;
};

/*!
 * \brief Implementation of the learning-to-rank objective function, GY_NDCG
 * [arxiv.org/abs/xxxx].
 */
class RankGYNDCG : public RankingObjective {
  public:
  explicit RankGYNDCG(const Config& config) : RankingObjective(config) {}

  explicit RankGYNDCG(const std::vector<std::string>& strs)
      : RankingObjective(strs) {}

  ~RankGYNDCG() {}

  void Init(const Metadata& metadata, data_size_t num_data) override {
    RankingObjective::Init(metadata, num_data);
    for (data_size_t i = 0; i < num_queries_; ++i) {
      rands_.emplace_back(seed_ + i);
    }
  }

  inline void GetGradientsForOneQuery(data_size_t query_id, data_size_t cnt,
                                      const label_t* label, const double* score,
                                      score_t* lambdas,
                                      score_t* hessians) const override {
    //添加代码
    // Skip groups with too few items.
    if (cnt <= 1) {
      for (data_size_t i = 0; i < cnt; ++i) {
        lambdas[i] = 0.0f;
        hessians[i] = 0.0f;
      }
      return;
    }
    // Turn scores into a probability distribution using Softmax.
    std::vector<double> rho(cnt, 0.0);
    Common::Softmax(score, rho.data(), cnt);

    // An auxiliary buffer of parameters used to form the ground-truth
    // distribution and compute the loss.
    std::vector<double> params(cnt);
    // 定义数字
    double sum_l2 = 0.008;

    for (data_size_t i = 0; i < cnt; ++i) {
      params[i] = Phi(label[i], rands_[query_id].NextFloat());
    }

    for (data_size_t i = 0; i < cnt; ++i) {
      params[i] = 0.02;
    }

    for (data_size_t i = 0; i < cnt; ++i) {
      lambdas[i] += static_cast<score_t>(rho[i] * (sum_l2 - params[i]));
      hessians[i] = static_cast<score_t>(rho[i] * (1.0 - rho[i]));
    }
  
  };



  double Phi(const label_t l, double g) const {
    return Common::Pow(2, static_cast<int>(l)) - g;
  }
  const char* GetName() const override { return "rank_gyndcg"; }

 private:
  mutable std::vector<Random> rands_;
};


}  // namespace LightGBMbrew install cmake
#endif  // LightGBM_OBJECTIVE_RANK_OBJECTIVE_HPP_
