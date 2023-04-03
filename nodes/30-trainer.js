module.exports = function(RED) {
  function PropCompleted(config) {
    RED.nodes.createNode(this,config);

    var node = this;
    var cfg = config;

    node.stop_training = false;
    node.done_training = false;

    /*
     * The logic here is that we count how many output nodes there are and
     * multipe that number by the number of input nodes. Each output nodes
     * generates an event when it's done its reset. We collect these events
     * and count how many we received. Once we have gotten the input times
     * output number of events, we trigger a reset-completed event on this
     * node. That triggers a fast-forward on the network.
     *
     * In truth, input times output is baseline number of events. There are
     * many more depending on the connections within the network. Messages
     * are copied and spread out over  the network, so there are many many
     * more potential "reset-complete" events from the output nodes.
     */
    node.trigger_reset = () => {
      var msg = {}
      msg.resetid = RED.util.generateId();
      msg.topic = 'reset';

      var outnodescntr = 0;
      RED.nodes.eachNode( function(n) {
        if ( n.z == node.z && n.type == 'nnb-output-node' && n.d == undefined ) {
          outnodescntr += 1
        }
      });

      var cntr = {};
      for ( var idx = 0 ; idx < cfg.scope.length; idx++ ) {
        cntr[cfg.scope[idx]] = outnodescntr;
      }

      var hndlr = function(objmsg) {
        cntr[objmsg.input_node_id] -= 1

        var flag = true;
        for ( var idx = 0 ; idx < cfg.scope.length; idx++ ) {
          flag = flag && (cntr[cfg.scope[idx]] < 1)
        }

        if ( flag ) {
          node.off( 'reset-complete:' + objmsg.resetid, hndlr );
          node.emit('input', { topic: 'reset-completed' });
        }
      }
      node.on('reset-complete:'+msg.resetid, hndlr);

      /* send reset message to all input nodes */
      for ( var idx = 0 ; idx < cfg.scope.length; idx++ ) {
        RED.nodes.getNode( cfg.scope[idx] ).emit('input', msg );
      }
    };

    node.on('close', function() {
      node.status({});
    });

    /*
     * Deal with input messages.
    */
    node.on("input",function(msg, send, done) {
      if ( msg.topic == "dataset" ) {
        node.trainingset_index = 0;
        node.training_size     = parseInt(msg.training_size);

        node.testingset_index  = 0;
        node.testing_size      = parseInt(msg.testing_size);

        node.dataset           = msg.payload;
        node.stop_training     = false
        node.done_training     = false;

        node.trigger_reset()
        return;
      }

      if ( msg.topic == "backprop-completed" || msg.topic == "training-done" ) {
        /* trigger a network reset - this allows doing an evaluation of a
           data point.
         */
        node.trigger_reset();
        return;
      }

      if ( msg.topic == "output-generated" ) {
        if ( node.done_training ) {
          send({ ...msg,
                 topic: "test-output",
                 idx: node.testingset_index
          });
          node.trigger_reset();
        } else {
          /* intention that when this is connected to the backprop
             node, this will cause an training loop since the backprop generates
             this message when it has collected it's data for doing a backprop.
           */
          send({...msg, topic: "trigger-backprop"});
          done();
        }

        return;
      }

      if ( msg.topic == "random-weights" ) {
        for ( var idx = 0 ; idx < cfg.scope.length; idx++ ) {
          msg = { ...msg,
                  max: parseFloat(cfg.weightmax),
                  min: parseFloat(cfg.weightmin),
          };
          RED.nodes.getNode( cfg.scope[idx] ).emit('input', msg );
        }
        return;
      }

      /* Capture reset and send datapoint. For each computation of the network
         it needs to be reset, be for back propagation or testing the test
         data (see below). Hence once the reset has gone through the entire
         network and come out the other end, a new datapoint can be pushed into
         the network.
       */
      if ( msg.topic == "reset-completed" && node.done_training == false ) {
        if ( (node.trainingset_index < node.training_size) && !node.stop_training ) {
          var datapoint = node.dataset[node.trainingset_index];

          for ( var idx = 0 ; idx < cfg.scope.length; idx++ ) {
            var nde = RED.nodes.getNode( cfg.scope[idx] );

            nde.emit('input', { ...msg,
                                topic: "",
                                value: datapoint[nde.name],
                                dp: {...datapoint}
            });
          }

          node.trainingset_index += 1;

          node.status( {
            fill:"green",
            shape:"dot",
            text:"training " + node.trainingset_index + " / " + node.training_size
          });
        } else {
          node.status( {
            fill:"green",
            shape:"dot",
            text:"training done"
          });

          node.trainingset_index = 0;
          node.testingset_index = 0;
          node.done_training = true;

          send({ topic: "training-done" });
          done();
        }

        return;
      }

      /* this is the reset completed but for handling testing data */
      if ( msg.topic == "reset-completed" && node.done_training == true ) {
        if ((node.testingset_index + node.training_size) > node.dataset.length ||
            (node.testingset_index >= node.testing_size) ) {

          setTimeout( () => { node.status({}) }, 450);
          node.done_training = false;
          node.stop_training = false;
          done();

          return;
        }

        var datapoint = node.dataset[node.testingset_index +
                                     node.training_size];

        for ( var ndIdx = 0 ; ndIdx < cfg.scope.length; ndIdx++ ) {
          var nde = RED.nodes.getNode( cfg.scope[ndIdx] );

          nde.emit('input', { ...msg,
                              topic: "",
                              value: datapoint[nde.name],
                              dp: {...datapoint}
          });
        }

        node.testingset_index++;

        node.status( {
          fill:"blue",
          shape:"dot",
          text:"testing " + node.testingset_index + " / " + node.testing_size
        });
      }

      /*
         training can't be halted since the backend is far faster than the
         frontend.
       */
      if ( msg.topic == "halt" ) {
        node.stop_training = true
        return;
      }
    });
  }

  RED.nodes.registerType("nnb-trainer",PropCompleted);
}
