import tensorflow as tf
from load_mnist import load_mnist,load_inv_size_mnist,load_firstfold_GPS_inv_size
from keras import backend as K
import numpy as np
import pickle
import random
import sys
import math
from CNNTFClass import training_CNN
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



# data_sets=load_inv_size_mnist('data',size_divisor=10) # data_sets.train.x.shape = 550
# data_sets=load_firstfold_GPS_inv_size('GPS',size_divisor=1) # data_sets.train.x.shape = 550
filename="paper2_data_for_DL_kfold_dataset_RL.pickle"
with open(filename, 'rb') as f:
  kfold_dataset, _ = pickle.load(f)

kth_fold=0
prop=1.0
num_class=5
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

# random_seed=0  
# np.random.seed(random_seed)
# Train_X=np.array(kfold_dataset[0][0])# len ---> 19287 --> shape:(19287,1,284,4)
# Train_labels=np.array(kfold_dataset[0][1]) # len ---> 19287: (0,1,2,3,4) labels
# Test_X=np.array(kfold_dataset[0][2])    # len ---> 4822 ---> shape: (4822,1,248,4)
# Test_labels=np.array(kfold_dataset[0][4]) # len ---> 4822 (0,1,2,3,4)  
# input_dim=data_sets.train.x.shape[1] # input_dim=784
test_accuracy, f1_macro, f1_weight,modelParam = training_CNN(Train_X,Train_Y,Train_Y_ori,
                                                  Val_X,Val_Y,Val_Y_ori,
                                                  Test_X,Test_Y,Test_Y_ori,
                                                  batch_size,input_size, seed=7,prop=prop, num_filter=[32])
print(len(modelParam[0]))
print(np.array(modelParam).shape)
exit()
# def variable(name, shape, initializer):
#     dtype = tf.float32
#     var = tf.get_variable(
#         name, 
#         shape, 
#         initializer=initializer, 
#         dtype=dtype)
#     return var

# def variable_with_weight_decay(name, shape, stddev, wd):
#     dtype = tf.float32
#     var = variable(
#         name, 
#         shape, 
#         initializer=tf.truncated_normal_initializer(
#             stddev=stddev, 
#             dtype=dtype))
 
#     if wd is not None:
#       weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
#       tf.add_to_collection('losses', weight_decay)

#     return var

# def normalize_vector(v):
#     norm_val = np.linalg.norm(np.concatenate(v))
#     norm_v = [a/norm_val for a in v]
#     return norm_v, norm_val

def get_vec_to_list_fn():
    params_val = sess.run(params)
    num_params = len(np.concatenate(params_val))        
    print('Total number of parameters: %s' % num_params)
    def vec_to_list(v):
        return_list = []
        cur_pos = 0
        for p in params_val:
            return_list.append(v[cur_pos : cur_pos+len(p)])
            cur_pos += len(p)

        assert cur_pos == len(v)
        return return_list

    return vec_to_list


# np.random.seed(0)
# tf.set_random_seed(0)
# if not os.path.exists(train_dir):
#     os.makedirs(train_dir)

# config=tf.ConfigProto()
# sess=tf.Session(config=config)
# K.set_session(sess)

# # Setup input
# input_placeholder=tf.placeholder(tf.float32,shape=(None,input_dim),name='input_placeholder')
# labels_placeholder=tf.placeholder(tf.int32,shape=(None),name='labels_placeholder')

# num_train_examples=data_sets.train.labels.shape[0]
# num_test_examples=data_sets.test.labels.shape[0]


# ################################## INFERENCE ##############################
# with tf.variable_scope('softmax_linear'):
#     weights=variable_with_weight_decay('weights',[input_dim*num_classes],
#                                 stddev=1.0/math.sqrt(float(input_dim)),wd=weight_decay)
#     logits=tf.matmul(input_placeholder,tf.reshape(weights,[input_dim,num_classes]))


#################################### LOSS #########################
# labels = tf.one_hot(labels_placeholder, depth=num_classes)    
# cross_entropy = - tf.reduce_sum(tf.multiply(labels, tf.nn.log_softmax(logits)), reduction_indices=1)
# indiv_loss_no_reg = cross_entropy
# loss_no_reg = tf.reduce_mean(cross_entropy, name='xentropy_mean')
# tf.add_to_collection('losses', loss_no_reg)
# total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

# global_step = tf.Variable(0, name='global_step', trainable=False)
# learning_rate = tf.Variable(initial_learning_rate, name='learning_rate', trainable=False)
# learning_rate_placeholder = tf.placeholder(tf.float32)
# update_learning_rate_op = tf.assign(learning_rate, learning_rate_placeholder)




# ################################### Optimizer #############################
# optim_Adam = tf.train.AdamOptimizer(learning_rate)
# train_opAdam = optim_Adam.minimize(total_loss, global_step=global_step)

# optim_SGD = tf.train.GradientDescentOptimizer(learning_rate)
# train_opSGD = optim_SGD.minimize(total_loss, global_step=global_step)

# correct = tf.nn.in_top_k(logits, labels_placeholder, 1)
# accuracy_op=tf.reduce_sum(tf.cast(correct, tf.int32)) / tf.shape(labels_placeholder)[0]

# preds = tf.nn.softmax(logits, name='preds')  

# all_params = []
# for layer in ['softmax_linear']:
#     for var_name in ['weights']:                
#         temp_tensor = tf.get_default_graph().get_tensor_by_name("%s/%s:0" % (layer, var_name))            
#         all_params.append(temp_tensor)      
# params=all_params

grad_total_loss_op = tf.gradients(total_loss, params)
grad_loss_no_reg_op =tf.gradients(loss_no_reg,params)
v_placeholder = [tf.placeholder(tf.float32, shape=a.get_shape()) for a in params]
u_placeholder = [tf.placeholder(tf.float32, shape=a.get_shape()) for a in params]
hessian_vector = hessian_vector_product(total_loss, params, v_placeholder)
grad_loss_wrt_input_op = tf.gradients(total_loss, input_placeholder)        

influence_op = tf.add_n(
    [tf.reduce_sum(tf.multiply(a, array_ops.stop_gradient(b))) for a, b in zip(grad_total_loss_op, v_placeholder)])

grad_influence_wrt_input_op = tf.gradients(influence_op, input_placeholder)
# checkpoint_file = os.path.join(train_dir, "%s-checkpoint" % model_name)

# # Here I have reduced number of datapoints learned in the train().
# all_train_feed_dict = {input_placeholder: data_sets.train.x,labels_placeholder: data_sets.train.labels}
# all_test_feed_dict = {input_placeholder: data_sets.test.x,labels_placeholder: data_sets.test.labels}

# init = tf.global_variables_initializer()        
# sess.run(init)




W_placeholder = tf.placeholder(tf.float32,shape=[input_dim * num_classes],name='W_placeholder')
set_weights = tf.assign(weights, W_placeholder, validate_shape=True)
set_params_op= [set_weights]

######################################### TRAIN ################################

# def train( num_steps=None, 
#           iter_to_switch_to_batch=None, 
#           iter_to_switch_to_sgd=None,
#           save_checkpoints=True, verbose=True):
#     X_train = all_train_feed_dict[input_placeholder]
#     Y_train = all_train_feed_dict[labels_placeholder]
#     num_train_examples = len(Y_train)
#     assert len(Y_train.shape) == 1
#     assert X_train.shape[0] == Y_train.shape[0]

#     if num_train_examples == num_train_examples:
#         if verbose: print('Using normal model')
#         model = sklearn_model
#     elif num_train_examples == num_train_examples - 1:
#         if verbose: print('Using model minus one')
#         model = sklearn_model_minus_one
#     else:
#         raise ValueError 

#     model.fit(X_train, Y_train)
#     W = np.reshape(model.coef_.T, -1)

#     params_feed_dict = {}
#     params_feed_dict[W_placeholder] = W

#     sess.run(set_params_op, feed_dict=params_feed_dict)    
#     sess.run(logits,feed_dict=all_train_feed_dict)
#     if save_checkpoints: saver.save(sess, checkpoint_file, global_step=0)

#     if verbose:
#         print('LBFGS training took %s iter.' % model.n_iter_)
#         print('After training with LBFGS: ')    

# train()
##############################################################################
exit()


data_sets=load_firstfold_GPS_inv_size(Train_X,Train_Y,Val_X,Val_Y,Test_X,Test_Y,'GPS',size_divisor=1) # data_sets.train.x.shape = 550
test_idx=8

def test_retraining(test_idx, iter_to_load=0, force_refresh=False, 
                    num_to_remove=5, num_steps=1000, random_seed=0,remove_type='maxinf'):
    np.random.seed(0)
    load_checkpoint(0)

    # Predicted Loss
    y_test=data_sets.test.labels[test_idx]
    predicted_loss_diffs=get_influence_on_test_loss(
        [test_idx],
        np.arange(len(data_sets.train.labels)))
    
    indices_to_remove=np.argsort(np.abs(predicted_loss_diffs))[-num_to_remove:]
    predicted_loss_diffs=predicted_loss_diffs[indices_to_remove]

    
    # Actual Loss
    actual_loss_diffs=np.zeros([num_to_remove])
    
        # Sanity check    
    test_input_feed1 = data_sets.test.x[test_idx, :].reshape(1, -1)
    test_labels_feed1 = data_sets.test.labels[test_idx].reshape(-1)
    
    test_feed_dict = {input_placeholder: test_input_feed1,labels_placeholder: test_labels_feed1,}
    test_loss_val, params_val = sess.run([loss_no_reg, params], feed_dict=test_feed_dict)
    train_loss_val = sess.run(total_loss, feed_dict=all_train_feed_dict)

    # Retrain
    for step in range(num_steps):   
        sess.run(train_opAdam, feed_dict=all_train_feed_dict)    
    retrained_test_loss_val = sess.run(loss_no_reg, feed_dict=test_feed_dict)
    retrained_train_loss_val = sess.run(total_loss, feed_dict=all_train_feed_dict)
    print('Sanity check: what happens if you train the model a bit more?')
    print('Loss on test idx with original model    : %s' % test_loss_val)
    print('Loss on test idx with retrained model   : %s' % retrained_test_loss_val)
    print('Difference in test loss after retraining     : %s' % (retrained_test_loss_val - test_loss_val))
    print('===')
    print('Total loss on training set with original model    : %s' % train_loss_val)
    print('Total loss on training with retrained model   : %s' % retrained_train_loss_val)
    print('Difference in train loss after retraining     : %s' % (retrained_train_loss_val - train_loss_val))
    print('These differences should be close to 0.\n')

    # Retraining experiment
    for counter,idx_to_remove in enumerate(indices_to_remove):
        print("===#%s===" % counter)
        print('Retraining without train_idx %s (label %s):' % (idx_to_remove, data_sets.train.labels[idx_to_remove]))

        num_examples = data_sets.train.x.shape[0]
        idx = np.array([True] * num_examples, dtype=bool)
        idx[idx_to_remove] = False
        train_feed_dict = {input_placeholder: data_sets.train.x[idx, :],labels_placeholder: data_sets.train.labels[idx]}
        # retrain
        for step in range(num_steps):   
            sess.run(train_opAdam, feed_dict=train_feed_dict)    
        retrained_test_loss_val, retrained_params_val = sess.run([loss_no_reg, params], feed_dict=test_feed_dict)
        actual_loss_diffs[counter] = retrained_test_loss_val - test_loss_val        

        print('Diff in params: %s' % np.linalg.norm(np.concatenate(params_val) - np.concatenate(retrained_params_val)))      
        print('Loss on test idx with original model    : %s' % test_loss_val)
        print('Loss on test idx with retrained model   : %s' % retrained_test_loss_val)
        print('Difference in loss after retraining     : %s' % actual_loss_diffs[counter])
        print('Predicted difference in loss (influence): %s' % predicted_loss_diffs[counter])
        load_checkpoint(iter_to_load,do_checks=False)

    np.savez(
        'output/%s_loss_diffs' % model_name, 
        actual_loss_diffs=actual_loss_diffs, 
        predicted_loss_diffs=predicted_loss_diffs)

    print('Correlation is %s' % pearsonr(actual_loss_diffs, predicted_loss_diffs)[0])
    return actual_loss_diffs, predicted_loss_diffs, indices_to_remove








def get_influence_on_test_loss(test_indices,train_idx,approx_type='cg'):
    
    op=grad_loss_no_reg_op
    batch_size=100    
    num_iter=int(np.ceil(len(test_indices)/batch_size)) # its value=1
    start=0
    end=1
    input_feed_test=data_sets.test.x[test_indices[start:end],:].reshape(len(test_indices[start:end]),-1)
    labels_feed_test=data_sets.test.labels[test_indices[start:end]].reshape(-1)

    test_feed_dict={input_placeholder: input_feed_test,labels_placeholder: labels_feed_test,}

    temp=sess.run(op,feed_dict=test_feed_dict)
    test_grad_loss_no_reg_val=[a * (end-start) for a in temp]
    test_grad_loss_no_reg_val = [a for a in test_grad_loss_no_reg_val]

    
    print('Norm of test gradient: %s' % np.linalg.norm(np.concatenate(test_grad_loss_no_reg_val)))

    start_time=time.time()
    
    test_description=test_indices

    
    approx_filename = os.path.join(train_dir, '%s-%s-%s-test-%s.npz' % (model_name, 'cg', 'normal_loss', test_description))    

    if os.path.exists(approx_filename):
        inverse_hvp = list(np.load(approx_filename)['inverse_hvp'])
        print('Loaded inverse HVP from %s' % approx_filename)
    else:
        inverse_hvp = get_inverse_hvp_cg(test_grad_loss_no_reg_val,verbose=True)

        np.savez(approx_filename, inverse_hvp=inverse_hvp)
        print('Saved inverse HVP to %s' % approx_filename)

    inverse_hvp2=np.array(inverse_hvp).reshape([input_dim,num_classes])

    duration = time.time() - start_time
    # print('Inverse HVP took %s sec' % duration)

    start_time = time.time()   
            # it calculates influence funciton for EACH trining datapoint cuz we want to pick up those points
            # which have highest values of IF which is delta L. So predicted_loss_diffs[] length is equal to 
            # training set. At the end the highest values of IF as "maxinf" will be considered for removing.
    num_to_remove=len(train_idx)
    predicted_loss_diffs=np.zeros([num_to_remove])
    for counter,idx_to_remove in enumerate(train_idx):
        temp_input_feed=data_sets.train.x[idx_to_remove, :].reshape(1, -1)
        temp_labels_feed=data_sets.train.labels[idx_to_remove].reshape(-1)

        single_train_feed_dict={input_placeholder: temp_input_feed,labels_placeholder: temp_labels_feed,}
        train_grad_loss_val=sess.run(grad_total_loss_op, feed_dict=single_train_feed_dict)

        predicted_loss_diffs[counter] = np.dot(np.concatenate(inverse_hvp), np.concatenate(train_grad_loss_val)) / num_train_examples
    duration=time.time()-start_time
    return predicted_loss_diffs            #len(predicted_loss_diffs)=training_set




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
        feed_dict = fill_feed_dict_with_batch(data_sets.train, batch_size=batch_size)
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


def fill_feed_dict_with_batch(data_set, batch_size=0):
    if batch_size is None:
        feed_dict_with_batch = {input_placeholder: data_set.x,
                                labels_placeholder: data_set.labels}
        return feed_dict_with_batch
    elif batch_size == 0:
        batch_size = batch_size
    input_feed, labels_feed = data_set.next_batch(batch_size)                              
    feed_dict = {input_placeholder: input_feed,
                    labels_placeholder: labels_feed,}
    return feed_dict            


def load_checkpoint(iter_to_load, do_checks=True):
    checkpoint_to_load = "%s-%s" % (checkpoint_file, iter_to_load) 
    saver.restore(sess, checkpoint_to_load)

    if do_checks:
        print('Model %s loaded. Sanity checks ---' % checkpoint_to_load)


actual_loss_diffs, predicted_loss_diffs_cg, indices_to_remove = test_retraining(
    test_idx,
    iter_to_load=0,
    force_refresh=False,
    num_to_remove=10,
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
# axs[1].scatter(actual_loss_diffs, predicted_loss_diffs_lissa, zorder=2)
# axs[1].set_title('Linear (approx)', fontsize=fontsize)
print(actual_loss_diffs)
print(predicted_loss_diffs_cg)
plt.show()


