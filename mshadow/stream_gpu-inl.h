/*!
 *  Copyright (c) 2014 by Contributors
 * \file stream_gpu-inl.h
 * \brief implementation of GPU code
 * \author Bing Xu, Tianqi Chen
 */
#ifndef MSHADOW_STREAM_GPU_INL_H_
#define MSHADOW_STREAM_GPU_INL_H_
#include "./base.h"
#include "./tensor.h"
#include "./logging.h"

namespace mshadow {
#if MSHADOW_USE_CUDA == 1
// Stream alocation
// actual implementation of GPU stream in CUDA
template<>
struct Stream<gpu> {
  /*! \brief handle state */
  enum HandleState {
    NoHandle = 0,
    OwnHandle = 1,
  };
  /*! \brief cudaStream */
  hipStream_t stream_;
  /*! \brief rocblas handle */
  rocblas_handle blas_handle_;
  /*! \brief cudnn handle */
  #if MSHADOW_USE_CUDNN == 1
 miopenHandle_t dnn_handle_;
  #endif
  /*! \brief rocblas handle ownership */
  HandleState blas_handle_ownership_;
  /*! \brief cudnn handle ownership */
  HandleState dnn_handle_ownership_;
  /*! \brief cudaDeviceProp */
  hipDeviceProp_t prop;
  /*! \brief dev id */
  int dev_id;

  Stream(void) : stream_(0),
                 blas_handle_ownership_(NoHandle),
                 dnn_handle_ownership_(NoHandle) {}
  /*!
   * \brief wait for all the computation associated
   *  with this stream to complete
   */
  inline void Wait(void) {
    MSHADOW_CUDA_CALL(hipStreamSynchronize(stream_));
  }
  /*!
   * \brief query whether the the stream is idle
   * \return true if the stream is idle and all the job have been completed
   */
  inline bool CheckIdle(void) {
    hipError_t err = hipStreamQuery(stream_);
    if (err == hipSuccess) return true;
    if (err == hipErrorNotReady) return false;
    LOG(FATAL) << hipGetErrorString(err);
    return false;
  }
  /*!
   * \brief returns actual cudaStream_t given an input GPU stream pointer
   * \param stream pointer to GPU stream
   */
  inline static hipStream_t GetStream(Stream<gpu> *stream) {
    if (stream == NULL) {
#if MSHADOW_FORCE_STREAM
      LOG(FATAL) << "Default GPU stream was used when MSHADOW_FORCE_STREAM was on";
#endif
      return 0;
    } else {
      return stream->stream_;
    }
  }
  /*!
   * \brief return actual hipblasHandle
   * \param pointer to GPU stream
   */
  inline static rocblas_handle GetBlasHandle(Stream<gpu> *stream) {
    if (stream == NULL) {
      return 0;
    } else {
      CHECK_NE(stream->blas_handle_ownership_, NoHandle)
        << "No handle exist in source stream";
      return stream->blas_handle_;
    }
  }
  /*! \brief Destory rocblas handle if own it */
  inline void DestoryBlasHandle() {
    if (blas_handle_ownership_ == OwnHandle) {
      rocblas_status err = rocblas_destroy_handle(blas_handle_);
      blas_handle_ownership_ = NoHandle;
      CHECK_EQ(err, rocblas_status_success) << "Destory rocblas.handle failed";
    }
  }
  /*! \brief Destory original blas handle and create a new one */
  inline void CreateBlasHandle() {
    this->DestoryBlasHandle();
    rocblas_status err = rocblas_create_handle(&blas_handle_);
    blas_handle_ownership_ = OwnHandle;
    CHECK_EQ(err, rocblas_status_success) << "Create rocblas.handle failed";
  }
// #if MSHADOW_USE_CUDNN && defined(__HIPCC__)
#if MSHADOW_USE_CUDNN == 1
  inline static miopenHandle_t GetDnnHandle(Stream<gpu> *stream) {
    if (stream == NULL) {
      return 0;
    } else {
      CHECK_NE(stream->dnn_handle_ownership_, NoHandle) << "No handle exist in source stream";
      return stream->dnn_handle_;
    }
  }
#endif
  inline void DestroyDnnHandle() {
// #if MSHADOW_USE_CUDNN && defined(__HIPCC__)
#if MSHADOW_USE_CUDNN == 1
    if (dnn_handle_ownership_ == OwnHandle) {
      miopenStatus_t  err = miopenDestroy(dnn_handle_);
      CHECK_EQ(err, miopenStatusSuccess) << (err);
    }
#endif
  }
  inline void CreateDnnHandle() {
// #if MSHADOW_USE_CUDNN == 1 && defined(__HIPCC__)
#if MSHADOW_USE_CUDNN == 1
    this->DestroyDnnHandle();
    miopenStatus_t  err = miopenCreate(&dnn_handle_);
    CHECK_EQ(err, miopenStatusSuccess) << (err);
    err = miopenSetStream(dnn_handle_, stream_);
    CHECK_EQ(err, miopenStatusSuccess) << (err);
    this->dnn_handle_ownership_ = OwnHandle;
#endif
  }
};
template<>
inline Stream<gpu> *NewStream<gpu>(bool create_blas_handle,
                                   bool create_dnn_handle,
                                   int dev_id) {
  Stream<gpu> *st = new Stream<gpu>();
  MSHADOW_CUDA_CALL(hipStreamCreate(&st->stream_));
  if (create_blas_handle) {
    st->CreateBlasHandle();
  }
  if (create_dnn_handle) {
    st->CreateDnnHandle();
  }
  st->dev_id = dev_id;
  if (dev_id != -1) {
    MSHADOW_CUDA_CALL(hipGetDeviceProperties(&st->prop, dev_id));
  }
  return st;
}
template<>
inline void DeleteStream<gpu>(Stream<gpu> *stream) {
  MSHADOW_CUDA_CALL(hipStreamDestroy(stream->stream_));
  stream->DestoryBlasHandle();
  stream->DestroyDnnHandle();
  delete stream;
}
#endif
}  // namespace mshadow
#endif  // MSHADOW_STREAM_GPU_INL_H_
