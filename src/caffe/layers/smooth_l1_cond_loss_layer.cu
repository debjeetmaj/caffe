#include <vector>

#include "caffe/layers/smooth_l1_cond_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void SmoothL1CondLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  cond_ = static_cast<int>(bottom[4]->cpu_data()[0]);
  if(cond_==0){
  SmoothL1LossLayer<Dtype>::Forward_gpu(bottom,top);
  }
  else{
    top[0]->mutable_cpu_data()[0] = 0;
  }
  
}

template <typename Dtype>
void SmoothL1CondLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) { 
    if(cond_==0)
      SmoothL1LossLayer<Dtype>::Backward_gpu(top,propagate_down,bottom);
    else{
     for (int i = 0; i < 2; ++i) {
        if (propagate_down[i])
             bottom[i]->mutable_cpu_diff()[0]=0;
      }
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(SmoothL1CondLossLayer);

}  // namespace caffe
