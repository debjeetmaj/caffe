#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_cond_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithCondLossLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    cond_ = static_cast<int>(bottom[2]->cpu_data()[0]);
  if(cond_==0){
  SoftmaxWithLossLayer<Dtype>::Forward_gpu(bottom,top);
  }
  else{
    top[0]->mutable_cpu_data()[0] = 0;
  }
}

template <typename Dtype>
__global__ void SoftmaxCondLossBackwardGPU(const int nthreads, Dtype* bottom_diff, 
                const int num, const int dim, const int spatial_dim) {
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
      for (int c = 0; c < channels; ++c) {
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
      }
  }
}

template <typename Dtype>
void SoftmaxWithCondLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    //backpropagate when cond_==0
    if(cond_==0)
      SoftmaxWithLossLayer<Dtype>::Backward_cpu(top,propagate_down,bottom);
    else{
      Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
      // const Dtype* label = bottom[1]->cpu_data();
      int dim =  SoftmaxWithLossLayer<Dtype>::prob_.count() / SoftmaxWithLossLayer<Dtype>::outer_num_;
      const int nthreads = SoftmaxWithLossLayer<Dtype>::outer_num_ * SoftmaxWithLossLayer<Dtype>::inner_num_;
      // NOLINT_NEXT_LINE(whitespace/operators)
    SoftmaxCondLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
        CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_diff,
        SoftmaxWithLossLayer<Dtype>::outer_num_, dim, SoftmaxWithLossLayer<Dtype>::inner_num_);

    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithCondLossLayer);

}  // namespace caffe
