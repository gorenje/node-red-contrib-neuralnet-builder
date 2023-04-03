module.exports = function(RED) {
  RED.nodes.registerType("nnb-layer-node",
                         require('./lib/common.js').neuralNode.HiddenNode(RED) );
}
