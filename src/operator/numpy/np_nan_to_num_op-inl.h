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
 * Copyright (c) 2019 by Contributors
 * \file np_nan_to_num_op-inl.h
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_NAN_TO_NUM_OP_H_
#define MXNET_OPERATOR_NUMPY_NP_NAN_TO_NUM_OP_H_


#include <vector>
#include <numeric>
#include <set>
#include <string>
#include "../operator_common.h"
#include "../contrib/boolean_mask-inl.h"

#include "../../common/utils.h"

#include <dmlc/optional.h>
#include <dmlc/parameter.h>
#include <mxnet/operator_util.h>
#include <utility>
#include <algorithm>
#include <climits>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../tensor/init_op.h"
#include <math.h>
using std::isinf;
using std::isnan;

namespace isinf_typed {
  template<typename DType>
  MSHADOW_XINLINE bool IsInf(volatile DType val) {
    return false;
  }
  template<>
  MSHADOW_XINLINE bool IsInf(volatile int val) {
    return isinf(val);
  }
  template<>
  MSHADOW_XINLINE bool IsInf(volatile float val) {
    return isinf(val);
  }
  template<>
  MSHADOW_XINLINE bool IsInf(volatile double val) {
    return isinf(val);
  }
  template<>
  MSHADOW_XINLINE bool IsInf(volatile long double val) {
    return isinf(val);
  }

  template<>
  MSHADOW_XINLINE bool IsInf(volatile mshadow::half::half_t val) {
    return (val.half_ & 0x7fff) >= 0x7c00;
  }
};  // namespace isinf_typed

namespace isnan_typed {
  template<typename DType>
  MSHADOW_XINLINE bool IsNan(volatile DType val) {
    return false;
  }
  template<>
  MSHADOW_XINLINE bool IsNan(volatile float val) {
    return isnan(val);
  }
  template<>
  MSHADOW_XINLINE bool IsNan(volatile double val) {
    return isnan(val);
  }
  template<>
  MSHADOW_XINLINE bool IsNan(volatile long double val) {
    return isnan(val);
  }

  template<>
  MSHADOW_XINLINE bool IsNan(volatile mshadow::half::half_t val) {
    return (val.half_ & 0x7fff) > 0x7c00;
  }
};  // namespace isnan_typed


namespace mxnet {
namespace op {

struct NumpyNanToNumParam : public dmlc::Parameter<NumpyNanToNumParam> {
  bool copy; 
  double nan;
  dmlc::optional<double> posinf, neginf; 
  DMLC_DECLARE_PARAMETER(NumpyNanToNumParam) {
    DMLC_DECLARE_FIELD(copy)
    .set_default(true)
    .describe("");
    DMLC_DECLARE_FIELD(nan)
    .set_default(0.0)
    .describe("");
    DMLC_DECLARE_FIELD(posinf)
    .set_default(dmlc::optional<double>())
    .describe("");
    DMLC_DECLARE_FIELD(neginf)
    .set_default(dmlc::optional<double>())
    .describe("");
  }
};

inline bool NumpyNanToNumOpType(const nnvm::NodeAttrs& attrs,
                            std::vector<int>* in_attrs,
                            std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  TYPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  TYPE_ASSIGN_CHECK(*in_attrs, 0, out_attrs->at(0));
  return out_attrs->at(0) != -1;
};

template<int req>
struct nan_to_num_forward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out_data, const DType* in_data, const DType nan, const DType posinf, const DType neginf) {

    DType val = in_data[i];
    if (isnan_typed::IsNan<DType>(val))  val = nan;
    if (val > 0 && isinf_typed::IsInf(val))  val = posinf;
    if (val < 0 && isinf_typed::IsInf(val))  val = neginf;
    KERNEL_ASSIGN(out_data[i], req, val);
  }
};

template<typename xpu>                                                        
void NumpyNanToNumOpForward(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  using namespace mxnet;



  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& in_data = inputs[0];
  const TBlob& out_data = outputs[0];
  const NumpyNanToNumParam& param = nnvm::get<NumpyNanToNumParam>(attrs.parsed);
  using namespace mxnet_op;

  if (!common::is_float(in_data.type_flag_)) return;
  
  MSHADOW_REAL_TYPE_SWITCH(out_data.type_flag_, DType, {
    DType defaultnan = static_cast<DType>(param.nan) ;
    DType posinf = (param.posinf.has_value()) ? static_cast<DType>(param.posinf.value()) : mshadow::red::limits::MaxValue<DType>();
    DType neginf = (param.neginf.has_value()) ? static_cast<DType>(param.neginf.value()) : mshadow::red::limits::MinValue<DType>();

    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<nan_to_num_forward<req_type>, xpu>::Launch(
          s, out_data.Size(), out_data.dptr<DType>(), in_data.dptr<DType>(),
          defaultnan, posinf, neginf);
    });
  });
}

template<int req>
struct nan_to_num_backward {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* in_grad, const DType* out_grad,
                                  const DType* in_data) {
    int val = 1*out_grad[i];
    if (isnan_typed::IsNan(in_data[i]))  val = 0;
    if (val > 0 && isinf_typed::IsInf(in_data[i]))  val = 0;
    if (val < 0 && isinf_typed::IsInf(in_data[i]))  val = 0;
    KERNEL_ASSIGN(in_grad[i], req, val);
  };
};

template<typename xpu>
void NumpyNanToNumOpBackward(const nnvm::NodeAttrs& attrs,
                         const OpContext& ctx,
                         const std::vector<TBlob>& inputs,
                         const std::vector<OpReqType>& req,
                         const std::vector<TBlob>& outputs) {
  CHECK_EQ(inputs.size(), 2U);
  CHECK_EQ(outputs.size(), 1U);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  const TBlob& out_grad = inputs[0];
  const TBlob& in_data = inputs[1];
  const TBlob& in_grad = outputs[0];
  using namespace mxnet_op;
  MSHADOW_TYPE_SWITCH(out_grad.type_flag_, DType, {
    MXNET_ASSIGN_REQ_SWITCH(req[0], req_type, {
      Kernel<nan_to_num_backward<req_type>, xpu>::Launch(
          s, in_grad.Size(), in_grad.dptr<DType>(), out_grad.dptr<DType>(),
          in_data.dptr<DType>());
    });
  });
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_NAN_TO_NUM_OP_H_
