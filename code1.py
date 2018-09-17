
def CNN_classifier(input_labeled,true_label,num_filter):    
    kernel_size = (1, 3)
    padding = 'same'
    strides = 1
    pool_size = (1, 2)
    conv_layer = input_labeled
    for i in range(len(num_filter)):
        scope_name = 'encoder_set_' + str(i + 1)
        with tf.variable_scope(scope_name):
            conv_layer = tf.layers.conv2d(inputs=conv_layer, activation=tf.nn.relu, filters=num_filter[i],
                                                name='conv_1', kernel_size=kernel_size, strides=strides,
                                                padding=padding)
        # if i % 2 != 0:
            conv_layer = tf.layers.max_pooling2d(conv_layer, pool_size=pool_size,
                                                          strides=pool_size, name='pool')

    dense = tf.layers.flatten(conv_layer)
    dense = tf.layers.dropout(dense, 0.5)
    classifier_output = tf.layers.dense(dense, num_class, name='output')

    # classifier_output = classifier_output(num_filter, input_labeled, num_dense)
    loss_cls = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_label, logits=classifier_output),
                              name='loss_cls')
    train_op = tf.train.AdamOptimizer().minimize(loss_cls)

    all_params = []
    for loop_name in ['encoder_set_'+str(i+1) for i in range(len(num_filter))]:
        for layer_name in [ 'conv_1']:
            for var_name in ['kernel', 'bias']:
                temp_tensor = tf.get_default_graph().get_tensor_by_name("%s/%s/%s:0" % (loop_name,layer_name, var_name))            
                all_params.append(temp_tensor)      
    for layer_name in [ 'output']:
        for var_name in ['kernel', 'bias']:
            temp_tensor = tf.get_default_graph().get_tensor_by_name("%s/%s:0" % (layer_name, var_name))            
            all_params.append(temp_tensor)          

    all_params.append(temp_tensor)
    return loss_cls,train_op,all_params

def Run_CNN(Train_X,Train_labels,num_filter,epochs=1):
    Train_Y = keras.utils.to_categorical(Train_labels, num_classes=num_class)
    input_size = (list(Train_X.shape)[1:])
    with tf.Session() as sess:
        input_labeled=tf.placeholder(tf.float32,shape=[None]+input_size,name='input_placeholder')
        true_label=tf.placeholder(tf.int32,shape=[None, num_class],name='labels_placeholder')
        loss_cls, optim,params = CNN_classifier(input_labeled, true_label, num_filter)        
        sess.run(tf.global_variables_initializer())
        # op = sess.graph.get_operations()
        # print([m.name for m in op])
        # print('-------------------------------------------------------')
        # print([m.name for m in op if 'output' in m.name])        
        # exit()

        num_batches = len(Train_X) // batch_size        
        # [-1, self.input_side, self.input_side, self.input_channels]
        for k in range(epochs):
            for i in range(num_batches):
                X_cls = Train_X[i * batch_size: (i + 1) * batch_size]
                Y_cls = Train_Y[i * batch_size: (i + 1) * batch_size]
                # print(X_cls.shape)
                # print(Y_cls.shape)
                # exit()               
                loss_cls_val, optim_val,model_params_val = sess.run([loss_cls, optim, params],feed_dict={input_labeled: X_cls, true_label: Y_cls})
                print('Epoch Num {}, Batches Num {}, Loss_cls {}'.format
                      (k, i, np.round(loss_cls_val, 3),))
