#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_cond_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithCondLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  SoftmaxWithLossLayer<Dtype>::LayerSetUp(bottom,top);
}

template <typename Dtype>
void SoftmaxWithCondLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  SoftmaxWithLossLayer<Dtype>::Reshape(bottom,top);
}

template <typename Dtype>
void SoftmaxWithCondLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  cond_ = static_cast<int>(bottom[2]->cpu_data()[0]);
  LOG(INFO) << "Cond  is" << cond_ ;
  if(cond_==0){
  SoftmaxWithLossLayer<Dtype>::Forward_cpu(bottom,top);
  }
  else{
    top[0]->mutable_cpu_data()[0] = 0;
  }
}

template <typename Dtype>
void SoftmaxWithCondLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[2]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    //backpropagate when cond_==0
    if(cond_==0)
      SoftmaxWithLossLayer<Dtype>::Backward_cpu(top,propagate_down,bottom);
    else{
      Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
      // const Dtype* label = bottom[1]->cpu_data();
      int dim =  SoftmaxWithLossLayer<Dtype>::prob_.count() / SoftmaxWithLossLayer<Dtype>::outer_num_;
      for(int i=0;i < SoftmaxWithLossLayer<Dtype>::outer_num_;i++){
        for(int j=0;j < SoftmaxWithLossLayer<Dtype>::inner_num_;j++){
          for (int c = 0; c < bottom[0]->shape(SoftmaxWithLossLayer<Dtype>::softmax_axis_); ++c) {
            bottom_diff[i * dim + c * SoftmaxWithLossLayer<Dtype>::inner_num_ + j] = 0;
          }
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithCondLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithCondLossLayer);
REGISTER_LAYER_CLASS(SoftmaxWithCondLoss);

}  // namespace caffe
