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
 * \file np_ediff1d-inl.h
 * \brief Function definition of numpy-compatible ediff1d operator
 */

#ifndef MXNET_OPERATOR_NUMPY_NP_EDIFF1D_OP_INL_H_
#define MXNET_OPERATOR_NUMPY_NP_EDIFF1D_OP_INL_H_

#include <mxnet/base.h>
#include <mxnet/operator_util.h>
#include <vector>
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"

namespace mxnet {
namespace op {

struct EDiff1DParam : public dmlc::Parameter<EDiff1DParam> {
  bool to_begin_arr_given, to_end_arr_given;
  dmlc::optional<double> to_begin_scalar;
  dmlc::optional<double> to_end_scalar;
  DMLC_DECLARE_PARAMETER(EDiff1DParam) {
    DMLC_DECLARE_FIELD(to_begin_arr_given).set_default(false).describe(
        "To determine whether the `to_begin` parameter is an array.");
    DMLC_DECLARE_FIELD(to_end_arr_given).set_default(false).describe(
        "To determine whether the `to_end` parameter is an array.");
    DMLC_DECLARE_FIELD(to_begin_scalar).set_default(dmlc::optional<double>()).describe(
        "If the `to_begin`is a scalar, the value of this parameter.");
    DMLC_DECLARE_FIELD(to_end_scalar).set_default(dmlc::optional<double>()).describe(
        "If the `to_end`is a scalar, the value of this parameter.");
  }
};

template<typename DType>
struct set_to_val {
  MSHADOW_XINLINE static void Map(index_t i, DType *out, double val) {
    out[i] = DType(val);
  }
};

template<int req>
struct ediff1d_forward {
  template <typename DType>
  MSHADOW_XINLINE static void Map(int i,
                                  DType* out_data,
                                  const DType* in_data,
                                  const index_t padding) {
    KERNEL_ASSIGN(out_data[i + padding], req, in_data[i + 1] - in_data[i]);
  }
};

template<int req>
struct ediff1d_backward_arr {
  template <typename DType>
  MSHADOW_XINLINE static void Map(int i,
                                  DType* igrad_dptr,
                                  const DType* input_dptr,
                                  const DType* ograd_dptr,
                                  const size_t padding,
                                  const size_t input_size) {
    if (i == 0) {
      KERNEL_ASSIGN(igrad_dptr[i], req, -ograd_dptr[i + padding]);
    } else if (i == input_size - 1) {
      KERNEL_ASSIGN(igrad_dptr[i], req, ograd_dptr[i - 1 + padding]);
    } else {
      KERNEL_ASSIGN(igrad_dptr[i], req, ograd_dptr[i - 1 + padding] - ograd_dptr[i + padding]);
    }
  }
};

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_NUMPY_NP_EDIFF1D_OP_INL_H_
