const nnb_utils = {
  statusBP: (node) => {
    node.status({fill:"blue",shape:"dot",text:"backprop"});
    setTimeout( () => { node.status({}); },450);
  },
  statusFW: (node) => {
    node.status({fill:"yellow",shape:"dot",text:"forward"});
    setTimeout( () => { node.status({}); },450);
  },
  statusRW: (node) => {
    node.status({fill:"grey",shape:"dot",text:"randomised weights"});
    setTimeout( () => { node.status({}); },450);
  },
  statusFired: (node,val) => {
    node.status({fill:"green",shape:"dot",text:"fired - " + val});
    setTimeout( () => {
      node.status({fill:"green", shape:"ring", text:("" + val)})
    }, 450);
  },
  statusNotFired: (node,val) => {
    node.status({fill:"red",shape:"dot",text:"not fired - " + val});
    setTimeout( () => {
      node.status({fill:"red", shape:"ring", text:("" + val)})
    },450);
  },
  commNewWeight: (nde,to,newWeight,red) => {
    red.comms.publish("neuralnet:" + nde, red.util.encodeObject({
      msg:    "weight-change",
      node:   nde,
      to:     to,
      weight: newWeight
    }));
  },
  actFuncts: require("./actfunctions.js"),

  /* magic constants from the trainer node */
  learningRate: (node,red) => {
    var rVal = 0.3;

    red.nodes.eachNode( function(n) {
      // z --> same flow, d --> true if disabled, else undefined.
      if (n.z == node.z && n.type == 'nnb-trainer' && n.d == undefined) {
        rVal = parseFloat(n.learningrate);
        return;
      }
    });

    return rVal;
  },

  clampWeight: (weight,node,red) => {
    var rVal = weight;

    red.nodes.eachNode( function(n) {
      // z --> same flow, d --> true if disabled, else undefined.
      if (n.z == node.z && n.type == 'nnb-trainer' && n.d == undefined) {
        rVal = Math.max( parseFloat(n.weightmin),
                         Math.min( parseFloat(n.weightmax), weight ) );
        return;
      }
    });

    return rVal;
  },
};

function NeuralNodes() {
  const nodeCache = (function(){
    let registry = {
      wiresOut: {},
      wiresIn: {},
      weights: {},
    };

    return {
      addNode(id,wires) {
        registry.wiresOut[id] = wires;

        for ( var idx = 0; idx < wires.length; idx+=1 ) {
          (registry.wiresIn[wires[idx]] ||= []).push( id );
        }
        return undefined;
      },
      countWiresIn(id) {
        return (registry.wiresIn[id] || []).length;
      },
      wiresIn(id) {
        return registry.wiresIn[id] || [];
      },
      weightFor(nodeId, prevNodeId) {
        if ( registry.weights[ nodeId ] && registry.weights[ nodeId ][ prevNodeId ] != undefined ) {
          /* done to ensure that 0.0 is also a possible value */
          return registry.weights[ nodeId ][ prevNodeId ];
        } else {
          return 1.0;
        }
      },
      setWeightFor(nodeId, prevNodeId, weight ) {
        (registry.weights[ nodeId ] ||= {})[ prevNodeId ] = weight;
      },
      reset() {
        registry = {
          wiresOut: {},
          wiresIn: {},
          weights: {},
        };
      }
    }
  })();

  return {

    InputNode: (RED) => {
      function OnMessageNeuralNode(config) {
        RED.nodes.createNode(this, config);

        var node = this;
        var cfg = config;

        nodeCache.addNode( node.id, node.wires[0] );

        node.backprop = (msg) => {
          /* here endth the backprogation */
          nnb_utils.statusBP(node);
        };

        node.on('close', function() {
          nodeCache.reset();
        });

        node.on('input', function(msg,send,done) {
          msg ||= {};

          if ( msg.topic == "reset" ) {
            send({ ...msg, input_node_id: node.id });
            if ( done ) ( done() );
            return;
          }

          if ( msg.topic == "random-weights" ) {
            nnb_utils.statusRW(node);
            send(msg);
            done();
            return;
          }
          nnb_utils.statusFW(node);

          msg.upstream_node_id = node.id;
          msg.value = msg.value;
          msg.fired = "yes";

          nnb_utils.statusFired(node,msg.value);
          send(msg);
          if ( done ) ( done() );
        });
      };

      return OnMessageNeuralNode;
    },


    HiddenNode: (RED) => {
      function OnMessageNeuralNode(config) {
        RED.nodes.createNode(this, config);

        var node = this;
        var cfg = config;

        nodeCache.addNode( node.id, node.wires[0] );

        node.trigger_count = nodeCache.countWiresIn(node.id);
        node.values = [];

        /*
         * Backpropagation computation for hidden neuron.
         *
         * This goes backwards through the network, it starts with
         * the output neurons and passes backwards towards the input
         * nodes.
         *
         * Because nodes will have backprop triggered multiple times, this
         * is an improvised backpropagation algorithm. A proper backprop
         * algorithm takes a holistic view of the network and considers
         * the entire network. That is not the case in this situation since
         * messages are being passed from one node to the other. This makes
         * the whole thing slightly random since it also depends on the
         * order in which backprop(...) method is called.
         */
        node.backprop = (msg) => {
          if ( node.values.length == 0 ) { return };

          nnb_utils.statusBP(node);

          if ( msg.nodetype == "output" || msg.nodetype == "hidden" ) {
            for ( var idx = 0; idx < node.values.length ; idx++ ) {
              if ( node.values[idx].fired == "no" ) { continue; }

              var upstream_node = RED.nodes.getNode(
                node.values[idx].upstream_node_id
              )

              var weightDelta = (
                node.values[idx].weight *
                ( msg.nodetype == "hidden" ? msg.value_percent : msg.weight_percent ) *
                (node.values[idx].value / msg.prev_value) *
                nnb_utils.learningRate(node,RED) *
                ( node.values[idx].value > msg.prev_value ? -1 : 1 )
              );

              var newWeight = nnb_utils.clampWeight(
                node.values[idx].weight + weightDelta,
                node,
                RED
              );

              nodeCache.setWeightFor(node.id, upstream_node.id, newWeight);
              nnb_utils.commNewWeight(node.id, upstream_node.id, newWeight, RED);

              upstream_node.backprop({
                ...msg,
                nodetype: "hidden",
                value_percent: node.values[idx].value / msg.prev_value,
                weight_delta: weightDelta,
                prev_value:  node.values[idx].value,
                prev_weight: node.values[idx].weight,
              });
            }
          }
        };

        /*
         * Flow restart.
         */
        node.on('close', function() {
          nodeCache.reset();
        });

        /*
         * Fast forward, reset and weight setting.
         *
         * This goes forward through the network and therefore uses
         * Node-RED send to pass messages between neurons.
         */
        node.on('input', function(msg,send,done) {
          msg ||= {};

          var t = {};

          if ( msg.topic == "reset" ) {
            node.trigger_count = nodeCache.countWiresIn(node.id);
            node.values = [];

            send(msg);
            if ( done ) ( done() );

            return;
          }

          if ( msg.topic == "random-weights" ) {
            var inWires = nodeCache.wiresIn(node.id);

            if ( inWires.length == 0 ) { return };

            for ( var idx = 0; idx < inWires.length ; idx++ ) {
              var newWeight = (Math.random() * (msg.max-msg.min)) + msg.min;
              nodeCache.setWeightFor( node.id, inWires[idx], newWeight);
              nnb_utils.commNewWeight( node.id, inWires[idx], newWeight, RED);
            }

            nnb_utils.statusRW(node);

            send(msg);
            done();
            return;
          }

          nnb_utils.statusFW(node);

          node.trigger_count = node.trigger_count - 1;
          node.values.push( {
            upstream_node_id: msg.upstream_node_id,
            weight: nodeCache.weightFor( node.id, msg.upstream_node_id ),
            value: msg.value,
            fired: msg.fired
          });

          if ( node.trigger_count == 0 ) {
            var value = 0;
            var didSomethingFire = false;

            for ( var idx = 0 ; idx < node.values.length ; idx+=1 ) {
              var v = node.values[idx];

              if ( v.fired == "yes" ) {
                didSomethingFire = true;
                value += (v.weight * v.value);
              }
            }

            msg.value = nnb_utils.actFuncts[cfg.actfunct]( parseFloat(cfg.bias) + value );
            msg.upstream_node_id = node.id;

            if ( msg.value > cfg.threshold && didSomethingFire) {
              msg.fired = "yes";
              nnb_utils.statusFired(node,msg.value);
            } else {
              msg.fired = "no";
              nnb_utils.statusNotFired(node,msg.value);
            }

            send(msg);
            if ( done ) ( done() );
          }
        });
      };

      return OnMessageNeuralNode;
    },


    OutputNode: (RED) => {
      function OnMessageNeuralNode(config) {
        RED.nodes.createNode(this, config);

        var node = this;
        var cfg = config;

        nodeCache.addNode( node.id, node.wires[0] );

        node.trigger_count = 0;

        /*
         * Backpropagation computation for output neuron.
         *
         * This goes backwards through the network, it starts with
         * the output neurons and passes backwards towards the input
         * nodes. The input nodes then signal the completion of the
         * backpropagation.
         */
        node.backprop = function(msg) {
          var inWires = nodeCache.wiresIn(node.id);

          if ( inWires.length == 0 ) { return };

          nnb_utils.statusBP(node);

          var weights_total = 0;
          for ( var idx = 0; idx < node.values.length ; idx++ ) {
            if ( node.values[idx].fired == "no" ) {
              continue;
            }
            weights_total += node.values[idx].weight;
          }

          /*
           * avoid division by zero errors - take a "random" value.
           */
          if ( weights_total == 0 ) {
            weights_total = inWires.length;
          }

          for ( var idx = 0; idx < node.values.length ; idx++ ) {
            if ( node.values[idx].fired == "no" ) {
              continue;
            }

            var upstream_node = RED.nodes.getNode(
              node.values[idx].upstream_node_id
            )

            var weightDelta = (msg.desired_value - msg.current_value) * (
              msg.current_value * (1 - msg.current_value)
            );

            var newWeight = nnb_utils.clampWeight(
              node.values[idx].weight + (nnb_utils.learningRate(node,RED) *
                                                                 weightDelta),
              node,
              RED
            );

            nodeCache.setWeightFor(node.id,upstream_node.id,newWeight);
            nnb_utils.commNewWeight(node.id,upstream_node.id,newWeight,RED);

            upstream_node.backprop({
              ...msg,
              nodetype:       "output",
              weight_delta:   weightDelta,
              prev_value:     node.values[idx].value,
              weight_percent: node.values[idx].weight / weights_total
            });
          }
        }

        node.on('close', function() {
          nodeCache.reset();
        });

        /*
         * Fast forward, reset and weight setting.
         *
         * This is goes forward through the network and therefore uses
         * Node-RED send to pass messages between neurons.
         */
        node.on('input', function(msg,send,done) {
          msg ||= {};

          /*
           * A reset message is sent through the network to reset all value
           * collectors. After this message, comes a fast-forward message,
           * followed by a backpropagation step and then the reset message ...
           * and so on.
           */
          if ( msg.topic == "reset" ) {
            /*
             * Trigger count is the number of messages this node should receive
             * before generating it's output. Nominally this is the number
             * of inWires however nodes might have been disabled, disabling
             * the input but not removing the wire.
             */
            node.trigger_count = 0;

            /*
             * Wires might exist but the connecting node might have
             * been disabled.
             */
            var inWires = nodeCache.wiresIn(node.id);
            for ( var idx = 0; idx < inWires.length; idx++ ) {
              var nd = RED.nodes.getNode( inWires[idx] );
              /* d -> disabled, if undefined, node is enabled */
              if ( nd && nd.d == undefined ) {
                node.trigger_count++;
              }
            }
            node.values = [];

            RED.nodes.eachNode( function(n) {
              // z --> same flow, d --> true if disabled, else undefined.
              if (n.z == node.z && n.type == 'nnb-trainer' && n.d == undefined) {
                RED.nodes.getNode( n.id ).emit('reset-complete:' + msg.resetid,
                                               msg );
              }
            });

            send(msg);
            done();

            return;
          }

          /*
           * Initialise the weights between neurons randomly. Bounds for
           * weights are defined in the message as 'max' and 'min'.
           */
          if ( msg.topic == "random-weights" ) {
            var inWires = nodeCache.wiresIn(node.id);

            if ( inWires.length == 0 ) { return };

            for ( var idx = 0; idx < inWires.length ; idx++ ) {
              var newWeight = (Math.random() * (msg.max - msg.min)) + msg.min;
              nodeCache.setWeightFor( node.id, inWires[idx], newWeight);
              nnb_utils.commNewWeight( node.id, inWires[idx], newWeight, RED);
            }

            nnb_utils.statusRW(node);
            return;
          }

          /*
           * This is the fast-forward step, that is generate a value for each
           * input. That becomes the current value for the propagation
           * step that follows after this.
           */
          node.trigger_count = node.trigger_count - 1;
          node.values.push( {
            upstream_node_id: msg.upstream_node_id,
            weight: nodeCache.weightFor(node.id, msg.upstream_node_id),
            value: msg.value,
            fired: msg.fired
          });
          nnb_utils.statusFW(node);

          if ( node.trigger_count == 0 ) {
            var value = 0;

            for ( var idx = 0 ; idx < node.values.length ; idx+=1 ) {
              var v = node.values[idx];

              if ( v.fired == "yes" ) {
                value += (v.weight * v.value);
              }
            }

            msg.value            = value;
            msg.desired_value    = msg.dp[node.name];
            msg.output_node_id   = node.id;
            msg.output_node_name = node.name;
            msg.topic            = "output";

            delete msg.upstream_node_id;
            delete msg.fired;

            nnb_utils.statusFired(node, "got: " + msg.value + " exp: " +
                                        msg.desired_value);
            send(msg);
            done();
          }
        });
      };

      return OnMessageNeuralNode;
    },
  };
};

/*
 * This setup is inspired from link-in/out nodes which also create a node
 * wide cache. This does the same but for separate nodes, i.e. link-in/out
 * define the nodes within the model, this is slightly different because
 * the nodes are defined in the .js files found above this directory.
 *
 * That's why this magic variable is defined here - so that the cache is the
 * singleton accross all types of neural net nodes.
 */
const magic = NeuralNodes();
module.exports.neuralNode = {
  InputNode: magic.InputNode,
  OutputNode: magic.OutputNode,
  HiddenNode: magic.HiddenNode
};
