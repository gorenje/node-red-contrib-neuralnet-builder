module.exports = function(RED) {
  function PropCompleted(config) {
    RED.nodes.createNode(this,config);

    var node = this;
    var cfg = config;

    node.values = [];

    node.on("input",function(msg, send, done) {
      if ( msg.topic == "reset" ) {
        node.values = [];
        return;
      }

      if ( msg.topic == "training-done" ) {
        /* Assume this is connected to the trainer node and just echo
         * the message back to it - it's generated by the trainer.
         */
        send(msg);
        done();
        return;
      }

      /*
       * Here the node expects as many 'output' messages as the scope, i.e.,
       * the output neurons it is connected to. Once all outputs have
       * arrived, a backpropagation may be triggered.
       */
      if ( msg.topic == "output" ) {
        if ( cfg.scope.indexOf(msg.output_node_id) < 0 ) {
          return;
        }

        node.values.push({ ...msg });

        if ( node.values.length == cfg.scope.length ) {
          send({ topic: "output-generated", values: node.values });
          done();
          return
        }

        return;
      }

      if ( msg.topic == "trigger-backprop" ) {

        if ( node.values.length < cfg.scope.length ) {
          send(msg);
          done();
          return;
        }

        send(msg);
        node.status({ fill:"green", shape:"dot", text:"started backprop" });

        var error_values = {
          "total": 0
        }
        for ( var idx = 0 ; idx < node.values.length ; idx++ ) {
          var outdata = node.values[idx];

          var node_error = Math.pow( outdata.desired_value -
                                     outdata.value, 2) * 0.5;

          error_values[outdata.output_node_id] = node_error;
          error_values["total"] += node_error;
        }

        for ( var idx = 0 ; idx < node.values.length ; idx++ ) {
          var output_nde = RED.nodes.getNode( node.values[idx].output_node_id );

          output_nde.backprop({
            current_value: node.values[idx].value,
            desired_value: node.values[idx].desired_value,
            error_values:  { ...error_values }
          });
        }

        node.status({});
        node.bp_call_counter = 0;
        node.values = [];
        send({topic: "backprop-completed"});
        done();
      }
    });
  }
  RED.nodes.registerType("nnb-backprop",PropCompleted);
}
