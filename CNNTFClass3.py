import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import random
import pickle
import tensorflow as tf
# from load_mnist import load_mnist,load_inv_size_mnist,load_firstfold_GPS_inv_size2
from keras import backend as K
import sys
import math
import os.path
import keras
from hessians import hessian_vector_product
from tensorflow.python.ops import array_ops
from sklearn import linear_model, preprocessing, cluster
from scipy.optimize import fmin_ncg
import seaborn as sns
import time
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import os





def train_val_split(Train_X, Train_Y_ori):
    val_index = []
    for i in range(num_class):
        label_index = np.where(Train_Y_ori == i)[0]
        val_index.append(label_index[:round(0.1*len(label_index))])
    val_index = np.hstack(tuple([label for label in val_index]))
    Val_X = Train_X[val_index]
    Val_Y_ori = Train_Y_ori[val_index]
    Val_Y = keras.utils.to_categorical(Val_Y_ori, num_classes=num_class)
    train_index_ = np.delete(np.arange(0, len(Train_Y_ori)), val_index)
    Train_X = Train_X[train_index_]
    Train_Y_ori = Train_Y_ori[train_index_]
    Train_Y = keras.utils.to_categorical(Train_Y_ori, num_classes=num_class)
    return Train_X, Train_Y, Train_Y_ori, Val_X, Val_Y, Val_Y_ori

filename="py2_data_for_DL_kfold_dataset_RL.pickle"
with open(filename, 'rb') as f:
  kfold_dataset, _ = pickle.load(f)

# Settings
batch_size = 100
latent_dim = 800
units = 800  # num unit in the MLP hidden layer
num_dense = 0
kernel_size = (1,3)
padding = 'same'
strides = 1
pool_size = (1, 3)
num_class = 5
num_filter=[32]
epochs=2
initializer = tf.glorot_uniform_initializer()
test_idx=8
test_indices=[test_idx]
kth_fold=0
prop=0.1
Train_X = kfold_dataset[kth_fold][0]
Train_Y_ori = kfold_dataset[kth_fold][1]
Test_X = kfold_dataset[kth_fold][2]
Test_Y = kfold_dataset[kth_fold][3]
Test_Y_ori = kfold_dataset[kth_fold][4]
random.seed(7)
np.random.seed(7)
random_sample = np.random.choice(len(Train_X), size=round(prop*len(Train_X)), replace=False, p=None)
Train_X = Train_X[random_sample]
Train_Y_ori = Train_Y_ori[random_sample]
Train_X, Train_Y, Train_Y_ori, Val_X, Val_Y, Val_Y_ori = train_val_split(Train_X, Train_Y_ori)
num_train_examples=np.shape(Train_X)[0]
# Train_X:                                                      ---->shape=(17358,1,248,,4)
# Train_Y: [[0.,0.,0.,1.],[0.,0.,0.,1.],[0.,1.,0.,0.] ....]     ---->shape=(17358,5)
# Train_Y_ori: [4,4,1, ....]                                    ---->shape=(17358,)
# Test_X:                                                       ---->shape=(4822,1,248,,4)             
# Test_Y: [[0.,1.,0.,0.],[0.,0.,1.,0.],[0.,1.,0.,0.] ....]      ---->shape=(4822,5)
# Test_Y_ori: [1,3,1, ....]                                     ---->shape=(4822,)
input_size = list(np.shape(Test_X)[1:])
#---------------------------------------------------------
weight_decay=0.01
batch_size=100
initial_learning_rate=0.001
keep_probs=None
max_lbfgs_iter=1000
mini_batch=False
train_dir='output'
log_dir='log'
damping=0.0
model_name='mnist_logreg_lbfgs'
tf.reset_default_graph()
sess= tf.Session()
input_labeled = tf.placeholder(dtype=tf.float32, shape=[None] + input_size, name='input_labeled')
true_label = tf.placeholder(tf.float32, shape=[None, num_class], name='true_label')

############################## CLASSIFIER #############################
conv_layer = input_labeled
for i in range(len(num_filter)):
    scope_name = 'encoder_set_' + str(i + 1)
    with tf.variable_scope(scope_name):
        conv_layer = tf.layers.conv2d(inputs=conv_layer, activation=tf.nn.relu, filters=num_filter[i],
                                            name='conv_1', kernel_size=kernel_size, strides=strides,
                                            padding=padding)
    if i % 2 != 0:
        conv_layer = tf.layers.max_pooling2d(conv_layer, pool_size=pool_size,
                                                      strides=pool_size, name='pool')
dense = tf.layers.flatten(conv_layer)
dense = tf.layers.dropout(dense, 0.5)
classifier_output = tf.layers.dense(dense, num_class, name='FC_4')
all_params = []
for loop_name in ['encoder_set_'+str(i+1) for i in range(len(num_filter))]:
    for layer_name in ['conv_1']:
        for var_name in ['kernel', 'bias']:
            temp_tensor = tf.get_default_graph().get_tensor_by_name("%s/%s/%s:0" % (loop_name,layer_name, var_name))            
            all_params.append(temp_tensor)      
params=all_params


# def cnn_model(input_labeled, true_label, num_filter):
# classifier_output,params,testq = classifier(num_filter, input_labeled, num_dense)

cross_entropy=tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_label, logits=classifier_output,name='cross_entropy')
                          

loss_cls = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_label, logits=classifier_output),
                          name='loss_cls')
train_op = tf.train.AdamOptimizer().minimize(loss_cls)

tf.add_to_collection('loss_cls', loss_cls)
total_loss = tf.add_n(tf.get_collection('loss_cls'), name='total_loss')    
gholi=tf.shape(total_loss)

correct_prediction = tf.equal(tf.argmax(true_label, 1), tf.argmax(classifier_output, 1))
accuracy_cls = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

grad_total_loss_op = tf.gradients(loss_cls, params)
grad_loss_no_reg_op =tf.gradients(loss_cls,params)


paramsHessian=tf.concat([tf.reshape(params[0],[-1]),tf.reshape(params[1],[-1])],axis=0)
paramsHessian2=[paramsHessian]
v_placeholder = [tf.placeholder(tf.float32, shape=a.get_shape()) for a in paramsHessian2]
# v_placeholder = [tf.placeholder(tf.float32, shape=paramsHessian)]
hessian_vector = hessian_vector_product(total_loss, paramsHessian2, v_placeholder)
# grad_loss_wrt_input_op = tf.gradients(total_loss, input_labeled)        

# influence_op = tf.add_n([tf.reduce_sum(tf.multiply(a, array_ops.stop_gradient(b))) for a, b in zip(grad_total_loss_op, v_placeholder)])
# grad_influence_wrt_input_op = tf.gradients(influence_op, input_labeled)
all_train_feed_dict = {input_labeled: Train_X,true_label:Train_Y}
all_test_feed_dict = {input_labeled: Test_X,  true_label:Test_Y}


def loss_acc_evaluation(Test_X, Test_Y, sess, input_labeled, true_label, k, loss_cls, accuracy_cls):
    metrics = []
    for i in range(len(Test_X) // batch_size):
        Test_X_batch = Test_X[i * batch_size:(i + 1) * batch_size]
        Test_Y_batch = Test_Y[i * batch_size:(i + 1) * batch_size]
        loss_cls_, accuracy_cls_ = sess.run([loss_cls, accuracy_cls],
                                            feed_dict={input_labeled: Test_X_batch,
                                                       true_label: Test_Y_batch})
        metrics.append([loss_cls_, accuracy_cls_])
    Test_X_batch = Test_X[(i + 1) * batch_size:]
    Test_Y_batch = Test_Y[(i + 1) * batch_size:]
    if len(Test_X_batch)>=1:
        loss_cls_, accuracy_cls_ = sess.run([loss_cls, accuracy_cls], feed_dict={input_labeled: Test_X_batch,
                                                   true_label: Test_Y_batch})
    metrics.append([loss_cls_, accuracy_cls_])
    mean_ = np.mean(np.array(metrics), axis=0)
    print('Epoch Num {}, Loss_cls_Val {}, Accuracy_Val {}'.format(k, mean_[0], mean_[1]))
    return mean_[0], mean_[1]

def prediction_prob(Test_X, classifier_output, input_labeled, sess):
    prediction = []
    for i in range(len(Test_X) // batch_size):
        Test_X_batch = Test_X[i * batch_size:(i + 1) * batch_size]
        prediction.append(sess.run(tf.nn.softmax(classifier_output), feed_dict={input_labeled: Test_X_batch}))
    Test_X_batch = Test_X[(i + 1) * batch_size:]
    if len(Test_X_batch) >= 1:
        prediction.append(sess.run(tf.nn.softmax(classifier_output), feed_dict={input_labeled: Test_X_batch}))
    prediction = np.vstack(tuple(prediction))
    y_pred = np.argmax(prediction, axis=1)
    return y_pred



# ======================================================================
val_accuracy = {-2: 0, -1: 0}
val_loss = {-2: 10, -1: 10}


# loss_cls, accuracy_cls, train_op,classifier_output,params,gholi,testq = cnn_model(input_labeled, true_label, num_filter)
sess.run(tf.global_variables_initializer())
# print(testq)
# exit()
# saver = tf.train.Saver(max_to_keep=20)
num_batches = len(Train_X) // batch_size
for k in range(epochs):
    for i in range(num_batches):      
        X_cls = Train_X[i * batch_size: (i + 1) * batch_size]
        Y_cls = Train_Y[i * batch_size: (i + 1) * batch_size]

        loss_cls_, accuracy_cls_,_, modelParam = sess.run([loss_cls, accuracy_cls, train_op,params],
                                               feed_dict={input_labeled: X_cls, true_label: Y_cls})            

    X_cls = Train_X[(i + 1) * batch_size:]
    Y_cls = Train_Y[(i + 1) * batch_size:]
    loss_cls_, accuracy_cls_,_, modelParam = sess.run([loss_cls, accuracy_cls, train_op,params],
                                           feed_dict={input_labeled: X_cls, true_label: Y_cls})
    print('Epoch Num {}, Batches Num {}, Loss_cls {}, Accuracy_train {}'.format
          (k, i, np.round(loss_cls_, 3), np.round(accuracy_cls_, 3)))
    print('====================================================')
    loss_val, acc_val = loss_acc_evaluation(Val_X, Val_Y, sess, input_labeled, true_label, k, loss_cls, accuracy_cls)
    val_loss.update({k: loss_val})
    val_accuracy.update({k: acc_val})
    print('====================================================')
    if not os.path.exists("Conv-Semi-TF-PS/" + str(prop)):
        os.makedirs("Conv-Semi-TF-PS/"  + str(prop))            
    
    # saver.save(sess, "Conv-Semi-TF-PS/" + str(prop), global_step=k)
    if all([val_accuracy[k] < val_accuracy[k - 1], val_accuracy[k] < val_accuracy[k - 2]]):
        break
print("Val Accuracy Over Epochs: ", val_accuracy)
print("Val Loss Over Epochs: ", val_loss)
max_accuracy_val = max(val_accuracy.items(), key=lambda k: k[1])
# saver.restore(sess, "Conv-Semi-TF-PS/" + str(prop) + '-' + str(max_accuracy_val[0]))

y_pred = prediction_prob(Test_X, classifier_output, input_labeled, sess)
test_acc = accuracy_score(Test_Y_ori, y_pred)
f1_macro = f1_score(Test_Y_ori, y_pred, average='macro')
f1_weight = f1_score(Test_Y_ori, y_pred, average='weighted')
print('CNN Classifier test accuracy {}'.format(test_acc))

############################ RETRAINING ###############################

def test_retraining(test_idx, iter_to_load=0, force_refresh=False, 
                    num_to_remove=5, num_steps=1000, random_seed=0,remove_type='maxinf'):
    np.random.seed(0)
    # load_checkpoint(0)

    # Predicted Loss
    y_test=Test_Y_ori

    predicted_loss_diffs=get_influence_on_test_loss(
        [test_idx],
        np.arange(len(Train_Y)))
    
    print("############################## Influence_Done ###############################")
    print("##############################################################################")    
    print("##############################################################################")

    indices_to_remove=np.argsort(np.abs(predicted_loss_diffs))[-num_to_remove:]
    predicted_loss_diffs=predicted_loss_diffs[indices_to_remove]

    
    # Actual Loss
    actual_loss_diffs=np.zeros([num_to_remove])
    
        # Sanity check    
    # test_input_feed1 = data_sets.test.x[test_idx, :].reshape(1, -1)
    # test_labels_feed1 = data_sets.test.labels[test_idx].reshape(-1)
    test_input_feed1=np.expand_dims(Test_X[test_indices[0],:], axis=0)   #(1,1,248,4)
    test_labels_feed1=Test_Y[test_indices[0]].reshape(1,5)

    test_feed_dict = {input_labeled: test_input_feed1,true_label: test_labels_feed1,}
    test_loss_val, params_val = sess.run([loss_cls, params], feed_dict=test_feed_dict)
    train_loss_val = sess.run(total_loss, feed_dict=all_train_feed_dict)

    print("Retrain=========================================")
    # for step in range(num_steps):   
    #     sess.run(train_op, feed_dict=all_train_feed_dict)    
    # retrained_test_loss_val = sess.run(loss_cls, feed_dict=test_feed_dict)
    # retrained_train_loss_val = sess.run(loss_cls, feed_dict=all_train_feed_dict)
    # print('Sanity check: what happens if you train the model a bit more?')
    # print('Loss on test idx with original model    : %s' % test_loss_val)
    # print('Loss on test idx with retrained model   : %s' % retrained_test_loss_val)
    # print('Difference in test loss after retraining     : %s' % (retrained_test_loss_val - test_loss_val))
    # print('===')
    # print('Total loss on training set with original model    : %s' % train_loss_val)
    # print('Total loss on training with retrained model   : %s' % retrained_train_loss_val)
    # print('Difference in train loss after retraining     : %s' % (retrained_train_loss_val - train_loss_val))
    # print('These differences should be close to 0.\n')

    # Retraining experiment
    for counter,idx_to_remove in enumerate(indices_to_remove):
        print("===#%s===" % counter)
        print('Retraining without train_idx %s (label %s):' % (idx_to_remove, Train_Y_ori[idx_to_remove]))

        num_examples = Train_X.shape[0]
        idx = np.array([True] * num_examples, dtype=bool)
        idx[idx_to_remove] = False
        train_feed_dict = {input_labeled: Train_X[idx, :],true_label: Train_Y[idx]}
        # retrain
        for step in range(num_steps):   
            sess.run(train_op, feed_dict=train_feed_dict)    
        retrained_test_loss_val, retrained_params_val = sess.run([loss_cls, params], feed_dict=test_feed_dict)
        actual_loss_diffs[counter] = retrained_test_loss_val - test_loss_val        
        print(np.shape(retrained_params_val))
        print(np.shape(params_val))  
        print('Diff in params: %s' % np.linalg.norm(np.array(params_val)- np.array(retrained_params_val)))      
        print('Loss on test idx with original model    : %s' % test_loss_val)
        print('Loss on test idx with retrained model   : %s' % retrained_test_loss_val)
        print('Difference in loss after retraining     : %s' % actual_loss_diffs[counter])
        print('Predicted difference in loss (influence): %s' % predicted_loss_diffs[counter])
        # load_checkpoint(iter_to_load,do_checks=False)

    # np.savez(
    #     'output/%s_loss_diffs' % model_name, 
    #     actual_loss_diffs=actual_loss_diffs, 
    #     predicted_loss_diffs=predicted_loss_diffs)

    print('Correlation is %s' % pearsonr(actual_loss_diffs, predicted_loss_diffs)[0])
    return actual_loss_diffs, predicted_loss_diffs, indices_to_remove


def get_influence_on_test_loss(test_indices,train_idx,approx_type='cg'):
    
    op=grad_loss_no_reg_op
    batch_size=100    
    input_feed_test=np.expand_dims(Test_X[test_indices[0],:], axis=0)   #(1,1,248,4)
    labels_feed_test=Test_Y[test_indices[0]].reshape(1,5)               #(1,5)

    test_feed_dict={input_labeled: input_feed_test,true_label: labels_feed_test,}
    temp=sess.run(op,feed_dict=test_feed_dict) 

    test_grad_loss_no_reg_val1=temp[0].reshape(-1)
    test_grad_loss_no_reg_val2=temp[1].reshape(-1)
    test_grad_loss_no_reg_val=np.concatenate((test_grad_loss_no_reg_val1,test_grad_loss_no_reg_val2)).reshape(1,-1) #(1,416)
    # test_grad_loss_no_reg_val=list(test_grad_loss_no_reg_val1)+list(test_grad_loss_no_reg_val2)            
    
    print('Norm of test gradient: %s' % np.linalg.norm(np.concatenate(test_grad_loss_no_reg_val)))
    start_time=time.time()
    
    test_description=test_indices

    
    approx_filename = os.path.join(train_dir, '%s-%s-%s-test-%s.npz' % (model_name, 'cg', 'normal_loss', test_description))    

    # if os.path.exists(approx_filename):
    #     inverse_hvp = list(np.load(approx_filename)['inverse_hvp'])
    #     print('Loaded inverse HVP from %s' % approx_filename)
    # else:
    inverse_hvp = get_inverse_hvp_cg(test_grad_loss_no_reg_val,verbose=True)
        # np.savez(approx_filename, inverse_hvp=inverse_hvp)
        # print('Saved inverse HVP to %s' % approx_filename)

    # inverse_hvp2=np.array(inverse_hvp).reshape([input_dim,num_classes])

    duration = time.time() - start_time
    # print('Inverse HVP took %s sec' % duration)

    start_time = time.time()   
            # it calculates influence funciton for EACH trining datapoint cuz we want to pick up those points
            # which have highest values of IF which is delta L. So predicted_loss_diffs[] length is equal to 
            # training set. At the end the highest values of IF as "maxinf" will be considered for removing.
    num_to_remove=len(train_idx)
    predicted_loss_diffs=np.zeros([num_to_remove])
    for counter,idx_to_remove in enumerate(train_idx):
        # temp_input_feed=Train_X[idx_to_remove, :].reshape(1, -1)
        # temp_labels_feed=Train_Y[idx_to_remove].reshape(-1)
        temp_input_feed=np.expand_dims(Train_X[idx_to_remove,:], axis=0)   #(1,1,248,4)
        temp_labels_feed=Train_Y[idx_to_remove].reshape(1,5)               #(1,5)

        single_train_feed_dict={input_labeled: temp_input_feed,true_label: temp_labels_feed,}
        train_grad_loss_val=sess.run(grad_total_loss_op, feed_dict=single_train_feed_dict)

        train_grad_loss_val2=np.concatenate((train_grad_loss_val[0].reshape(-1),train_grad_loss_val[1].reshape(-1)))
        # print(np.shape(train_grad_loss_val2))
        # print(np.shape(np.reshape(inverse_hvp,-1)))
        predicted_loss_diffs[counter] = np.dot(np.reshape(inverse_hvp,-1), train_grad_loss_val2) / num_train_examples
    duration=time.time()-start_time
    return predicted_loss_diffs            #len(predicted_loss_diffs)=training_set























def get_vec_to_list_fn():
    params2=tf.concat([tf.reshape(params[0],[-1]),tf.reshape(params[1],[-1])],axis=0)     # shape=(416,)
    params_val = sess.run(params2)                   # params shape:[[kernel[kernelwidth[numfilters],kernelheight[numfilters]],bias[numfilters]]]
    # case1=params_val[0].reshape(-1)
    # case2=params_val[1].reshape(-1)
    # params_val_list=np.concatenate((case1,case2))
    # params_val_list=[params_val_list]
    params_val_list=[params_val]
    num_params = len(params_val)
    print('Total number of parameters: %s' % num_params)
    def vec_to_list(v):
        return_list = []
        cur_pos = 0
        for p in params_val_list:
            return_list.append(v[cur_pos : cur_pos+len(p)])
            cur_pos += len(p)

        assert cur_pos == len(v)
        return return_list

    return vec_to_list



vec_to_list = get_vec_to_list_fn()

def get_inverse_hvp_cg(v, verbose):
    fmin_loss_fn = get_fmin_loss_fn(v)
    fmin_grad_fn = get_fmin_grad_fn(v)
    cg_callback = get_cg_callback(v, verbose)

    fmin_results = fmin_ncg(
        f=fmin_loss_fn,
        x0=np.concatenate(v),
        fprime=fmin_grad_fn,
        fhess_p=get_fmin_hvp,
        callback=cg_callback,
        avextol=1e-8,
        maxiter=100)

    return vec_to_list(fmin_results)
def get_fmin_hvp(x, p):
    hessian_vector_val = minibatch_hessian_vector_val(vec_to_list(p))

    return np.concatenate(hessian_vector_val)

def get_fmin_loss_fn(v):
    def get_fmin_loss(x):
        hessian_vector_val = minibatch_hessian_vector_val(vec_to_list(x))
        return 0.5 * np.dot(np.concatenate(hessian_vector_val), x) - np.dot(np.concatenate(v), x)
    return get_fmin_loss

def get_fmin_grad_fn(v):
    def get_fmin_grad(x):
        hessian_vector_val = minibatch_hessian_vector_val(vec_to_list(x))        
        return np.concatenate(hessian_vector_val) - np.concatenate(v)
    return get_fmin_grad

def get_cg_callback( v, verbose):
    fmin_loss_fn = get_fmin_loss_fn(v)
    
    def fmin_loss_split(x):
        hessian_vector_val = minibatch_hessian_vector_val(vec_to_list(x))
        return 0.5 * np.dot(np.concatenate(hessian_vector_val), x), -np.dot(np.concatenate(v), x)

def minibatch_hessian_vector_val(v):

    num_examples = num_train_examples
    if mini_batch == True:
        batch_size = 100
        assert num_examples % batch_size == 0
    else:
        batch_size = num_train_examples

    num_iter = int(num_examples / batch_size)
    # # reset dataset()
    # for data_set in data_sets:
    #     if data_set is not None:
    #         data_set.reset_batch()

    hessian_vector_val = None
    for i in range(num_iter):
        feed_dict = fill_feed_dict_with_batch(Train_X,Test_Y,i, batch_size=batch_size)
        # Can optimize this    
        feed_dict =update_feed_dict_with_v_placeholder(feed_dict, v)     
        hessian_vector_val_temp = sess.run(hessian_vector, feed_dict=feed_dict)

        if hessian_vector_val is None:
            hessian_vector_val = [b / float(num_iter) for b in hessian_vector_val_temp]
        else:
            hessian_vector_val = [a + (b / float(num_iter)) for (a,b) in zip(hessian_vector_val, hessian_vector_val_temp)]
    hessian_vector_val = [a + damping * b for (a,b) in zip(hessian_vector_val, v)]     
    return hessian_vector_val

def update_feed_dict_with_v_placeholder(feed_dict, vec):
    for pl_block, vec_block in zip(v_placeholder, vec):
        feed_dict[pl_block] = vec_block        
    return feed_dict

num_batches = len(Train_X) // batch_size
def fill_feed_dict_with_batch(Train_X,Train_Y,i, batch_size=0):
    global num_batches    
    assert i<num_batches    
    input_feed = Train_X[i * batch_size: (i + 1) * batch_size]
    labels_feed = Train_Y[i * batch_size: (i + 1) * batch_size]

    # input_feed, labels_feed = data_set.next_batch(batch_size)                              
    feed_dict = {input_labeled: input_feed,
                    true_label: labels_feed,}
    return feed_dict            

actual_loss_diffs, predicted_loss_diffs_cg, indices_to_remove = test_retraining(
    test_idx,
    iter_to_load=0,
    force_refresh=False,
    num_to_remove=2,
    remove_type='maxinf',
    random_seed=0)


actual_loss_diffs=actual_loss_diffs
predicted_loss_diffs_cg=predicted_loss_diffs_cg
# predicted_loss_diffs_lissa=predicted_loss_diffs_lissa
indices_to_remove=indices_to_remove

sns.set_style('white')
fontsize=16
fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(5, 1))

for ax in axs: 
    ax.set_aspect('equal')
    ax.set_xlabel('Actual diff in loss', fontsize=fontsize)
    ax.set_xticks(np.arange(-.0001, .0001, .0002))
    ax.set_yticks(np.arange(-.0001,.0001, .0002))
    ax.set_xlim([-.0001,.0001])
    ax.set_ylim([-.0001, .0001])
    ax.plot([-.0001,.0001], [-.0001, .0001], 'k-', alpha=0.2, zorder=1)
axs[0].set_ylabel('Predicted diff in loss', fontsize=fontsize)

axs[0].scatter(actual_loss_diffs, predicted_loss_diffs_cg, zorder=2)
axs[0].set_title('Linear (exact)', fontsize=fontsize)
print(actual_loss_diffs)
print(predicted_loss_diffs_cg)
plt.show()







