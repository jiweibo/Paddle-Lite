// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/core/mir/fusion/fc_gru_fuser.h"

#include <memory>
#include <vector>

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

void MulGruFuser::BuildPattern() {
  // create nodes.
  auto* x = VarNode("x")->assert_is_op_input("mul", "X");
  auto* w_i2h = VarNode("w_i2h")->assert_is_op_input("mul", "Y");
  auto* mul = OpNode("mul", "mul");
  auto* mul_out =
      VarNode("mul_out")->assert_is_op_output("mul")->assert_is_op_input(
          "gru", "Input");

  auto* w_h2h = VarNode("w_h2h")->assert_is_op_input("gru", "Weight");

  auto gru_teller = [](const Node* node) -> bool {
    std::string activation =
        const_cast<Node*>(node)->AsStmt().op_info()->GetAttr<std::string>(
            "activation");
    std::string gate_activation =
        const_cast<Node*>(node)->AsStmt().op_info()->GetAttr<std::string>(
            "gate_activation");
    bool has_attr_mode =
        const_cast<Node*>(node)->AsStmt().op_info()->HasAttr("origin_mode");
    return !has_attr_mode ||
           const_cast<Node*>(node)->AsStmt().op_info()->GetAttr<bool>(
               "origin_mode");
    return true && activation == "tanh" && gate_activation == "sigmoid";
  };

  auto* gru = OpNode("gru", "gru")->assert_node_satisfied(gru_teller);
  auto* batch_gate =
      VarNode("batch_gate")->assert_is_op_output("gru", "BatchGate");
  auto* batch_hidden =
      VarNode("batch_hidden")->assert_is_op_output("gru", "BatchHidden");
  auto* batch_reset_hidden_prev =
      VarNode("batch_reset_hidden_prev")
          ->assert_is_op_output("gru", "BatchResetHiddenPrev");
  auto* hidden = VarNode("hidden")->assert_is_op_output("gru", "Hidden");

  // create topology.
  std::vector<PMNode*> mul_inputs{w_i2h, x};
  std::vector<PMNode*> gru_inputs{mul_out, w_h2h};
  std::vector<PMNode*> gru_outputs{
      batch_gate, batch_hidden, batch_reset_hidden_prev, hidden};
  mul_inputs >> *mul >> *mul_out;

  // Some op specialities.
  mul_out->AsIntermediate();
  mul->AsIntermediate();
  gru->AsIntermediate();
  batch_gate->AsIntermediate();
  batch_hidden->AsIntermediate();
  batch_reset_hidden_prev->AsIntermediate();

  gru_inputs >> *gru >> gru_outputs;

  if (with_gru_bias_) {
    auto* b = VarNode("b")->assert_is_op_input("gru", "Bias");
    *b >> *gru;
  }
}

void MulGruFuser::InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) {
  auto op_desc = GenOpDesc(matched);
  auto fusion_gru_op = LiteOpRegistry::Global().Create("fusion_gru");
  auto gru = matched.at("gru")->stmt()->op();
  auto* scope = gru->scope();
  auto& valid_places = gru->valid_places();
  fusion_gru_op->Attach(op_desc, scope);

  auto* new_op_node =
      graph->GraphCreateInstructNode(fusion_gru_op, valid_places);

  IR_NODE_LINK_TO(matched.at("w_i2h"), new_op_node);
  IR_NODE_LINK_TO(matched.at("w_h2h"), new_op_node);
  IR_NODE_LINK_TO(matched.at("x"), new_op_node);
  IR_NODE_LINK_TO(new_op_node, matched.at("hidden"));
  if (with_gru_bias_) {
    IR_NODE_LINK_TO(matched.at("b"), new_op_node);
  }
}

cpp::OpDesc MulGruFuser::GenOpDesc(const key2nodes_t& matched) {
  auto op_desc = *matched.at("gru")->stmt()->op_info();

  op_desc.mutable_inputs()->clear();
  op_desc.mutable_outputs()->clear();
  op_desc.SetType("fusion_gru");
  op_desc.SetInput("Input", {matched.at("x")->arg()->name});
  op_desc.SetInput("Weight_i2h", {matched.at("w_i2h")->arg()->name});
  op_desc.SetInput("Weight_h2h", {matched.at("w_h2h")->arg()->name});
  if (with_gru_bias_) {
    op_desc.SetInput("Bias", {matched.at("b")->arg()->name});
  }
  op_desc.SetOutput("Hidden", {matched.at("hidden")->arg()->name});
  op_desc.SetAttr(
      "activation",
      matched.at("gru")->stmt()->op_info()->GetAttr<std::string>("activation"));
  op_desc.SetAttr("gate_activation",
                  matched.at("gru")->stmt()->op_info()->GetAttr<std::string>(
                      "gate_activation"));
  op_desc.SetAttr(
      "is_reverse",
      matched.at("gru")->stmt()->op_info()->GetAttr<bool>("is_reverse"));

  return op_desc;
}

}  // namespace fusion
}  // namespace mir
}  // namespace lite
}  // namespace paddle
