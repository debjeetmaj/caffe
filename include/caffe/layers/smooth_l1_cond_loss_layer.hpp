#ifndef CAFFE_SMOOTH_L1_COND_LOSS_LAYER_HPP_
#define CAFFE_SMOOTH_L1_COND_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/smooth_l1_loss_layer.hpp"

namespace caffe {

/**
 * @brief SmoothL1CondLossLayer
 *
 * Fast R-CNN for Adaptation
 * Written by Debjeet Majumdar
 */
template <typename Dtype>
class SmoothL1CondLossLayer : public SmoothL1LossLayer<Dtype> {
 public:
  explicit SmoothL1CondLossLayer(const LayerParameter& param)
      : SmoothL1LossLayer<Dtype>(param), cond_() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SmoothL1CondLoss"; }

  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 5; }

  /**
   * Unlike most loss layers, in the SmoothL1CondLossLayer we can backpropagate
   * to both inputs -- override to return true and always allow force_backward.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
    int cond_;
};

}  // namespace caffe

#endif  // CAFFE_SMOOTH_L1_COND_LOSS_LAYER_HPP_
