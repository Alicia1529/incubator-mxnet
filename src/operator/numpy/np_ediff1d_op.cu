/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file np_ediff1d_op.cu
 * \brief GPU implementation of numpy-compatible ediff1d operator
 */

#include "./np_ediff1d_op-inl.h"

namespace mxnet {
namespace op {

void EDiff1DForwardGPU(const nnvm::NodeAttrs& attrs,
                       const OpContext& ctx,
                       const std::vector<TBlob>& inputs,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  CHECK_GE(inputs.size(), 1U);
  CHECK_LE(inputs.size(), 3U);
  CHECK_EQ(req.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  const TBlob& in_data = inputs[0];
  const TBlob& out_data = outputs[0];
  mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
  MSHADOW_TYPE_SWITCH(out_data.type_flag_, DType, {
    const EDiff1DParam& param = nnvm::get<EDiff1DParam>(attrs.parsed);
    size_t padding = 0;
    size_t in_size = (in_data.Size() > 0)? in_data.Size() - 1: 0;
    index_t idx = 1;  // used to index the rest of input arrays

    if (param.to_begin_arr_given) {
      // if the `to_begin` parameter is an array, copy its values to the beginning of the out array
      CUDA_CALL(cudaMemcpyAsync(out_data.dptr<DType>(), inputs[idx].dptr<DType>(),
                                inputs[idx].Size() * sizeof(DType), cudaMemcpyDeviceToHost,
                                mshadow::Stream<gpu>::GetStream(s)));
      padding += inputs[idx].Size();
      idx += 1;
    } else if (param.to_begin_scalar.has_value()) {
      // if the `to_begin` parameter is a scalar, directly assign its value
      out_data.dptr<DType>()[0] = param.to_begin_scalar.value();
      padding += 1;
    }

    if (param.to_end_arr_given) {
      // if the `to_end` parameter is an array, copy its values to the end of the out array
      CUDA_CALL(cudaMemcpyAsync(out_data.dptr<DType>() + padding + in_size,
                                inputs[idx].dptr<DType>(),
                                inputs[idx].Size() * sizeof(DType),
                                cudaMemcpyDeviceToHost,
                                mshadow::Stream<gpu>::GetStream(s)));
    } else if (param.to_end_scalar.has_value()) {
      // if the `to_end` parameter is a scalar, directly assign its value
      out_data.dptr<DType>()[padding + in_size] = param.to_end_scalar.value();
    }

    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<ediff1d_forward<req_type>, gpu>::Launch(
        s, in_size, out_data.dptr<DType>(), in_data.dptr<DType>(), padding);
    });
  });
}

void EDiff1DBackwardGPU(const nnvm::NodeAttrs& attrs, const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  using namespace mshadow;
  using namespace mxnet_op;
  CHECK_GE(inputs.size(), 2U);
  CHECK_LE(inputs.size(), 4U);
  CHECK_GE(outputs.size(), 1U);
  CHECK_LE(outputs.size(), 3U);
  CHECK_EQ(req.size(), outputs.size());

  mshadow::Stream<gpu> *s = ctx.get_stream<gpu>();
  const EDiff1DParam& param = nnvm::get<EDiff1DParam>(attrs.parsed);

  const TBlob& ograd = inputs[0];
  const TBlob& input = inputs[1];
  const TBlob& igrad = outputs[0];
  size_t in_size = (input.Size() > 0)? input.Size() - 1: 0;

  MSHADOW_REAL_TYPE_SWITCH(ograd.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      size_t padding = 0;
      index_t idx = 1;  // start from the second argument of `outputs`
      if (param.to_begin_arr_given) {
        CUDA_CALL(cudaMemcpyAsync(outputs[idx].dptr<DType>(),
                                  ograd.dptr<DType>(),
                                  outputs[idx].Size() * sizeof(DType),
                                  cudaMemcpyDeviceToHost,
                                  mshadow::Stream<gpu>::GetStream(s)));
        padding += outputs[idx].Size();
        idx += 1;
      } else if (param.to_begin_scalar.has_value()) {
        padding += 1;
      }

      if (param.to_end_arr_given) {
        CUDA_CALL(cudaMemcpyAsync(outputs[idx].dptr<DType>(),
                                  ograd.dptr<DType>()+ in_size + padding,
                                  outputs[idx].Size() * sizeof(DType),
                                  cudaMemcpyDeviceToHost,
                                  mshadow::Stream<gpu>::GetStream(s)));
      }

      if (input.Size() == 0) return;
      if (input.Size() == 1) {
        Kernel<set_to_val<DType>, gpu>::Launch(s, 1, igrad.dptr<DType>(), 0);
      } else {
        Kernel<ediff1d_backward_arr<req_type>, gpu>::Launch(
          s, igrad.Size(), igrad.dptr<DType>(),
          input.dptr<DType>(), ograd.dptr<DType>(),
          padding, igrad.Size());
      }
    });
  });
}

NNVM_REGISTER_OP(_npi_ediff1d)
.set_attr<FCompute>("FCompute<gpu>", EDiff1DForwardGPU);

NNVM_REGISTER_OP(_npi_backward_ediff1d)
.set_attr<FCompute>("FCompute<gpu>", EDiff1DBackwardGPU);

}  // namespace op
}  // namespace mxnet
