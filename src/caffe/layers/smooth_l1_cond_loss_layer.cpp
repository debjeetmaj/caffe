#include <vector>

#include "caffe/layers/smooth_l1_cond_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void SmoothL1CondLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // SmoothL1LossLayer<Dtype>::LayerSetUp(bottom,top);
  SmoothL1LossParameter loss_param = this->layer_param_.smooth_l1_loss_param();
  SmoothL1LossLayer<Dtype>::sigma2_ = loss_param.sigma() * loss_param.sigma();
  SmoothL1LossLayer<Dtype>::has_weights_ = (bottom.size() >= 3);
  if (SmoothL1LossLayer<Dtype>::has_weights_) {
    CHECK_EQ(bottom.size(), 5) << "If weights are used, must specify both "
      "inside and outside weights";
  }
}

template <typename Dtype>
void SmoothL1CondLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  SmoothL1LossLayer<Dtype>::Reshape(bottom,top);
}

template <typename Dtype>
void SmoothL1CondLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void SmoothL1CondLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(SmoothL1CondLossLayer);
#endif

INSTANTIATE_CLASS(SmoothL1CondLossLayer);
REGISTER_LAYER_CLASS(SmoothL1CondLoss);

}  // namespace caffe
