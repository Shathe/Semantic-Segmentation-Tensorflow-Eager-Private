import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import os
import nets.Network as Segception
import nets.MiniNetv2 as MiniNetv2
import utils.Loader as Loader
from utils.utils import get_params, preprocess, lr_decay, convert_to_tensors, restore_state, init_model, get_metrics
import cv2
# enable eager mode
tf.enable_eager_execution()
tf.set_random_seed(7)
np.random.seed(7)


# Trains the model for certains epochs on a dataset
def train(loader, optimizer, model, epochs=5, batch_size=2, show_loss=False, augmenter=False, lr=None, init_lr=2e-4,
          saver=None, variables_to_optimize=None, evaluation=True, name_best_model = 'weights/best', preprocess_mode=None,
          aux_loss=False, labels_resize_factor=1, model_upsample_eval=1):
    training_samples = len(loader.image_train_list)
    steps_per_epoch = int(training_samples / batch_size) + 1
    best_miou = 0

    for epoch in range(epochs):  # for each epoch
        lr_decay(lr, init_lr, 1e-9, epoch, epochs - 1)  # compute the new lr
        print('epoch: ' + str(epoch) + '. Learning rate: ' + str(lr.numpy()))
        for step in range(steps_per_epoch):  # for every batch
            with tf.GradientTape() as g:
                # get batch
                x, y, mask = loader.get_batch(size=batch_size, train=True, augmenter=augmenter, labels_resize_factor=labels_resize_factor)
                x = preprocess(x, mode=preprocess_mode)
                [x, y, mask] = convert_to_tensors([x, y, mask])
                if aux_loss:
                    y_, aux_y_ = model(x, training=True, aux_loss=aux_loss, upsample=1)  # get output of the model
                    loss = tf.losses.softmax_cross_entropy(y, y_, weights=mask) +\
                           tf.losses.softmax_cross_entropy(y, aux_y_, weights=mask)  # compute loss
                else:
                    y_ = model(x, training=True, aux_loss=aux_loss, upsample=1)  # get output of the model
                    loss = tf.losses.softmax_cross_entropy(y, y_, weights=mask)  # compute loss

                if show_loss: print('Training loss: ' + str(loss.numpy()))

            # Gets gradients and applies them
            grads = g.gradient(loss, variables_to_optimize)
            optimizer.apply_gradients(zip(grads, variables_to_optimize))

        if evaluation:
            # get metrics
            #train_acc, train_miou = get_metrics(loader, model, loader.n_classes, train=True, preprocess_mode=preprocess_mode, labels_resize_factor=1, model_upsample=model_upsample_eval)
            test_acc, test_miou = get_metrics(loader, model, loader.n_classes, train=False, flip_inference=False,
                                              scales=[1], preprocess_mode=preprocess_mode, labels_resize_factor=1, model_upsample=model_upsample_eval)

            #print('Train accuracy: ' + str(train_acc.numpy()))
            #print('Train miou: ' + str(train_miou))
            print('Test accuracy: ' + str(test_acc.numpy()))
            print('Test miou: ' + str(test_miou))
            print('')

            # save model if bet
            if test_miou > best_miou:
                best_miou = test_miou
                saver.save(name_best_model)
        else:
              saver.save(name_best_model)

        loader.suffle_segmentation()  # sheffle trainign set


if __name__ == "__main__":
    # some parameters
    n_gpu = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = str(n_gpu)

    n_classes = 11
    batch_size = 6
    epochs = 0
    width = 960
    height = 720
    labels_resize_factor = 2
    channels = 3
    lr = 1e-3
    name_best_model = 'weights/camvid/best'
    dataset_path = 'Datasets/camvid'
    preprocess_mode = 'imagenet'  #possible values 'imagenet', 'normalize',None

    loader = Loader.Loader(dataFolderPath=dataset_path, n_classes=n_classes, problemType='segmentation',
                           width=width, height=height, channels=channels, median_frequency=0.0)

    # build model and optimizer
    model = Segception.ERFNet(num_classes=n_classes, weights='imagenet', input_shape=(None, None, channels))

    # optimizer
    learning_rate = tfe.Variable(lr)
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # Init models (optional, just for get_params function)
    init_model(model, input_shape=(batch_size, height, width, channels))

    variables_to_restore = model.variables
    variables_to_save = model.variables
    variables_to_optimize = model.variables

    # Init saver. can use also ckpt = tfe.Checkpoint((model=model, optimizer=optimizer,learning_rate=learning_rate, global_step=global_step)
    saver_model = tfe.Saver(var_list=variables_to_save)
    restore_model = tfe.Saver(var_list=variables_to_restore)

    # restore if model saved and show number of params
    restore_state(restore_model, name_best_model)
    get_params(model)

    train(loader=loader, optimizer=optimizer, model=model, epochs=epochs, batch_size=batch_size, augmenter='segmentation', lr=learning_rate,
          init_lr=lr, saver=saver_model, variables_to_optimize=variables_to_optimize, name_best_model=name_best_model,
          evaluation=True, aux_loss=False, preprocess_mode=preprocess_mode, labels_resize_factor=labels_resize_factor, model_upsample_eval=labels_resize_factor)

    # Test best model
    print('Testing model')
    test_acc, test_miou = get_metrics(loader, model, loader.n_classes, train=False, flip_inference=True, scales=[1],
                                      write_images=True, preprocess_mode=preprocess_mode, time_exect=True, labels_resize_factor=1, model_upsample=labels_resize_factor)
    print('Test accuracy: ' + str(test_acc.numpy()))
    print('Test miou: ' + str(test_miou))