import tensorflow as tf
import time
import sys
import numpy as np
import argparse
import os
# ===
import processing
import gene_hm
from load_data import load_batch_data
from nets.models import resnet_model_deconv
import config as cfg
import nets.model_config as mcfg


def cal_acc(output, gtMaps, batchSize):
    # 计算准确率
    def _argmax(tensor):
        resh = tf.reshape(tensor, [-1])
        argmax = tf.argmax(resh, 0)
        return (argmax // tensor.get_shape().as_list()[0], argmax % tensor.get_shape().as_list()[0])

    def _compute_err( u, v):
        u_x, u_y = _argmax(u)
        v_x, v_y =_argmax(v)
        return tf.divide(tf.sqrt(tf.square(tf.to_float(u_x - v_x)) + tf.square(tf.to_float(u_y - v_y))),
                         tf.to_float(91))

    def _accur(pred, gtMap, num_image):
        err = tf.to_float(0)
        for i in range(num_image):
            err = tf.add(err,_compute_err(pred[i], gtMap[i]))
        return tf.subtract(tf.to_float(1), err / num_image)

    joint_accur = []
    for i in range(cfg.NPOINTS):
        joint_accur.append(
            _accur(output[:, i,:, :], gtMaps[:, i,:, :],batchSize))
    return joint_accur


def train(path_save_model, path_save_log_train, path_save_log_val):
    print('==create save path==')
    if tf.gfile.Exists(path_save_model):
        tf.gfile.DeleteRecursively(path_save_model)
    if tf.gfile.Exists(path_save_log_train):
        tf.gfile.DeleteRecursively(path_save_log_train)
    if tf.gfile.Exists(path_save_log_val):
        tf.gfile.DeleteRecursively(path_save_log_val)
    tf.gfile.MakeDirs(path_save_model)
    tf.gfile.MakeDirs(path_save_log_train)
    tf.gfile.MakeDirs(path_save_log_val)

    path_finetune_model = cfg.SAVE_ROOT_PATH + '/model'

    print('==create save path done==')

    print(cfg.LR)
    global_step = tf.Variable(0, trainable=False)
    print('==create model==')
    with tf.name_scope('input'):
        input_image = tf.placeholder(tf.float32
                                     , [None, cfg.INPUT_WIDTH, cfg.INPUT_HEIGHT, cfg.INPUT_CHANNEL]
                                     , name='input_images')
        labels = tf.placeholder(tf.float32
                                , [None, cfg.NPOINTS, cfg.MODEL_OUTPUT_WIDTH, cfg.MODEL_OUTPUT_HEIGHT]
                                , name='labels')
        batch_size=tf.placeholder(tf.int32, None, name='batch_size')
    print('--input done--')

    with tf.name_scope('inference'):
        logits1 = resnet_model_deconv(input_image, batch_size)
    print('--inference done--')

    with tf.name_scope('loss'):
        diff1 = tf.subtract(logits1, labels)
        train_loss = tf.reduce_mean(tf.nn.l2_loss(diff1, name='l2loss'))
    print('--loss done--')

    # with tf.name_scope('accuracy'):
    #     joint_accur=cal_acc(logist2,labels,BATCH_SIZE)
    # with tf.name_scope('lr'):
    #     decay_lr=tf.train.exponential_decay(lr,global_step,decay_steps,decay_rate,staircase,name='learning_rate')#指数式衰减

    with tf.name_scope('saver'):
        saver = tf.train.Saver()

    with tf.name_scope('train'):
        with tf.name_scope('optimizer'):
            opti = tf.train.RMSPropOptimizer(cfg.LR)
        train_op = opti.minimize(train_loss, global_step=global_step)
    print('--optimizer done--')

    init = tf.global_variables_initializer()
    print('--init done--')

    tf.summary.scalar('loss', train_loss, collections=['train', 'test'])
    # tf.summary.scalar('learning_rate',decay_lr,collections=['train'])
    # for i in range(cfg.NPOINTS):
    #     tf.summary.scalar(str(i),joint_accur[i],collections=['train','test'])

    merged_summary_train = tf.summary.merge_all('train')
    merged_summary_test = tf.summary.merge_all('test')

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    config = tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options)
    config.gpu_options.allow_growth = True

    with tf.name_scope('train_batch'):
        batch_images, batch_labels = \
            load_batch_data.batch_samples(cfg.BATCH_SIZE, cfg.TRAIN_DATA_FILE_PATH, True)
    # with tf.name_scope('val_batch'):
    #     val_images, val_labels =
    #     load_batch_data.batch_samples(cfg.BATCH_SIZE, cfg.VAL_DATA_FILE_PATH, True)

    with tf.name_scope('Session'):
        sess = tf.Session(config=config)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)
        sess.run(init)
        ckpt = tf.train.get_checkpoint_state(path_finetune_model)
        if ckpt and ckpt.model_checkpoint_path:
            print('load model')
            saver.restore(sess, ckpt.model_checkpoint_path)
        train_writer = tf.summary.FileWriter(path_save_log_train,
                                             graph=tf.get_default_graph())
        val_writer = tf.summary.FileWriter(path_save_log_val)
        print('Start train')

        with tf.name_scope('training'):
            for epoch in range(cfg.TRAIN_EPOCHES):
                time_epoch_start = time.time()
                loss_total_epoch = 0.0
                print('* * * *第%d个Epoch* * * *' % (epoch+1))
                # beginTime = time.time()
                for i in range(cfg.EPOCH_SIZE):
                    data_images, data_labels = sess.run([batch_images, batch_labels])
                    # jiancha shifou wei kong
                    if np.any(np.isnan(data_images)):
                        print('no images')
                        continue
                    if np.any(np.isnan(data_labels)):
                        print('no label')
                        continue
                    norm_images = processing.image_normalization(data_images)  # 归一化图像
                    hm_labels = gene_hm.batch_gene_hm(data_labels)  # heatmap label
                    # xianshi shifou zhengque shengcheng label
                    # for i in range(cfg.BATCH_SIZE):
                    #     for j in range(cfg.NPOINTS):
                    #         print(np.max(hm_labels[i][j]))
                    #         print(hm_labels[i][j])
                    #     plt.imshow(norm_images[i])
                    #     plt.matshow(np.sum(hm_labels[i], axis=0))
                    #     plt.show()
                    train_step = sess.run(global_step)
                    if (i+1) % cfg.SAVE_CKPT_STEP == 0:
                        loss, summary = sess.run([train_loss, merged_summary_train],
                                                 feed_dict={input_image: norm_images, labels: hm_labels,
                                                            batch_size: cfg.BATCH_SIZE})

                        train_writer.add_summary(summary, train_step)
                        saver.save(sess, path_save_model+'/model%d.ckpt' % train_step)
                    else:
                        _, loss = sess.run([train_op, train_loss],
                                           feed_dict={input_image: norm_images, labels: hm_labels,
                                                      batch_size: cfg.BATCH_SIZE})

                    loss_total_epoch += loss

                    print('第%d个batch的loss%f' % (i+1, loss))
                print('* *第%d个epoch的loss%f* *' % (epoch + 1, loss_total_epoch))
                time_epoch_end = time.time()
                print('one epoch cost time:', str(time_epoch_end - time_epoch_start))
                print('* ' * 20)
                # if (epoch+1) % cfg.VAL_STEP == 0:
                #     time_val_start = time.time()
                #     loss_val_total_epoch = 0.0
                #     for j in range(cfg.VAL_EPOCH_SIZE):
                #         train_step = sess.run(global_step)
                #         data_val_images, data_val_labels = sess.run([val_images, val_labels])
                #         if np.any(np.isnan(data_val_images)):
                #             print('no images')
                #             continue
                #         if np.any(np.isnan(data_val_labels)):
                #             print('no label')
                #             continue
                #         val_norm_images = processing.image_normalization(data_val_images)  # 归一化图像
                #         val_hm_labels = gene_hm.batch_gene_hm(data_val_labels)  # heatmap label
                #         loss_val_step = sess.run(train_loss,
                #                              feed_dict={input_image: val_norm_images, labels: val_hm_labels,
                #                                         batch_size: cfg.BATCH_SIZE})
                #         print('val step loss: ', loss_val_step)
                #
                #         loss_val_total_epoch += loss_val_step
                #         if j == cfg.VAL_EPOCH_SIZE-1:
                #             val_summaries = sess.run(merged_summary_test,
                #                                      feed_dict={input_image: val_norm_images, labels: val_hm_labels,
                #                                                 batch_size: cfg.BATCH_SIZE})
                #             val_writer.add_summary(val_summaries, train_step)
                #
                #     loss_val_epoch = loss_val_total_epoch / cfg.VAL_EPOCH_SIZE
                #     print('val cost time:', str(time.time() - time_val_start))
                #     print('loss val epoch:', loss_val_epoch)
                #     print('* ' * 20)

            coord.request_stop()
            coord.join(threads)

        val_writer.flush()
        train_writer.flush()
        val_writer.close()
        train_writer.close()


def update_config(args):
    if args.batch_size:
        cfg.BATCH_SIZE = args.batch_size
    if args.input_width:
        cfg.INPUT_WIDTH = args.input_width
    if args.input_height:
        cfg.INPUT_HEIGHT = args.input_height
    if args.output_width:
        cfg.MODEL_OUTPUT_WIDTH = args.output_width
    if args.output_height:
        cfg.MODEL_OUTPUT_HEIGHT = args.output_height
    if args.epoch_size:
        cfg.EPOCH_SIZE = args.epoch_size
    if args.stack_times:
        mcfg.TIMES_STACK = args.stack_times
    try:
        cfg.SAVE_ROOT_PATH = args.save_path
    except Exception as info:
        print('save path is must',info)
        exit()

    print('Train model with params:\n',
          '--stack_times : %d\n' % mcfg.TIMES_STACK,
          '--input_width,input_height : %d %d\n' % (cfg.INPUT_WIDTH, cfg.INPUT_HEIGHT)
          )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stack_times',  type=int, help='hourglass stacked times')

    parser.add_argument('--input_width', type=int)
    parser.add_argument('--input_height', type=int)

    parser.add_argument('--output_width', type=int)
    parser.add_argument('--output_height', type=int)

    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--epoch_size', type=int)

    parser.add_argument('--save_path', type=str)
    args = parser.parse_args()
    update_config(args)
    path_save_config = cfg.SAVE_ROOT_PATH+'/CONFIG.TXT'
    path_save_model = cfg.SAVE_ROOT_PATH+'/model'
    path_save_log_train = cfg.SAVE_ROOT_PATH+'/log/train'
    path_save_log_val = cfg.SAVE_ROOT_PATH+'/log/val'

    if not os.path.exists(cfg.SAVE_ROOT_PATH):
        os.makedirs(cfg.SAVE_ROOT_PATH)

    with open(path_save_config, 'w') as f_save_info:
        cfg_dict = cfg.__dict__
        for key in sorted(cfg_dict.keys()):
            if key[0].isupper():
                cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                f_save_info.write(cfg_str)
        mcfg_dict = mcfg.__dict__
        for key in sorted(mcfg_dict.keys()):
            if key[0].isupper():
                cfg_str = '{}: {}\n'.format(key, mcfg_dict[key])
                f_save_info.write(cfg_str)
    try:
        train(path_save_model, path_save_log_train, path_save_log_val)
    except Exception as info:
        print(info)
        sys.exit()






# log_dir='E:/Project/stacked_hourglass_net/log'
# if tf.gfile.Exists(log_dir):
#     tf.gfile.DeleteRecursively(log_dir)
# tf.gfile.MakeDirs(log_dir)

if __name__ == '__main__':
    main()
