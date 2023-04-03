module.exports = function(RED) {
  RED.nodes.registerType("nnb-input-node",
                         require('./lib/common.js').neuralNode.InputNode(RED) );
}
