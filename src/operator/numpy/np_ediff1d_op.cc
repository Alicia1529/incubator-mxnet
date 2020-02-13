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
 * \file np_dediff1d_op.cc
 * \brief CPU implementation of numpy-compatible ediff1d operator
 */

#include "./np_ediff1d_op-inl.h"

namespace mxnet {
namespace op {


DMLC_REGISTER_PARAMETER(EDiff1DParam);

// NNVM_REGISTER_OP(_npi_ediff1d)
// .set_attr_parser(ParamParser<EDiff1DParam>)
// .set_num_inputs(
//   [](const nnvm::NodeAttrs& attrs) {
//     const NEDiff1DParam& param = nnvm::get<EDiff1DParam>(attrs.parsed);
//     int num_inputs = 1;
//     if (param.to_begin_arr_given) num_inputs += 1;
//     if (param.to_end_arr_given) num_inputs += 1;
//     return num_inputs;
//   })
// .set_num_outputs(1)
// .set_attr<nnvm::FListInputNames>("FListInputNames",
//   [](const NodeAttrs& attrs) {
//     const NEDiff1DParam& param = nnvm::get<EDiff1DParam>(attrs.parsed);
//     int num_inputs = 1;
//     if (param.to_begin_arr_given) num_inputs += 1;
//     if (param.to_end_arr_given) num_inputs += 1;
//     if (num_inputs == 1) return std::vector<std::string>{"input1"};
//     if (num_inputs == 2) return std::vector<std::string>{"input1", "input2"};
//     return std::vector<std::string>{"input1", "input2", "input3"};
//   })
// .set_attr<mxnet::FInferShape>("FInferShape",  EDiff1DOpShape)
// .set_attr<nnvm::FInferType>("FInferType", EDiff1DOpType)
// .set_attr<FCompute>("FCompute<cpu>", EDiff1DForward<cpu>)
// .set_attr<nnvm::FGradient>("FGradient", MakeZeroGradNodes)
// .set_attr<nnvm::FInplaceOption>("FInplaceOption",
//   [](const NodeAttrs& attrs) {
//     return std::vector<std::pair<int, int> >{{0, 0}};
//   })
// .add_arguments(EDiff1DParam::__FIELDS__());

// what is this
// .add_argument("a", "NDArray-or-Symbol", "Input ndarray")


}  // namespace op
}  // namespace mxnet
