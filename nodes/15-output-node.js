module.exports = function(RED) {
  RED.nodes.registerType("nnb-output-node",
                         require('./lib/common.js').neuralNode.OutputNode(RED) );
}
