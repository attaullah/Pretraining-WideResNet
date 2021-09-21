from absl import app
from absl import logging
import os
import sys
import signal
import datetime
from absl import flags
from train_utils import get_model, get_accuracy, dump_weights, start_training
from utils import program_duration
from dataset import get_dataset

FLAGS = flags.FLAGS


def setup_flags():
    """
    Commandline arguments
    """
    flags.DEFINE_enum(name="d", default="imagenet_resized/32x32", enum_values=["imagenet_resized/32x32",
                      "imagenet_resized/64x64", "imagenet-full"], help="dataset ")
    flags.DEFINE_string("n", help="network", default="wrn-28-2")
    flags.DEFINE_integer('bs', help="batch_size", default=128)
    flags.DEFINE_enum(name="lt", default="cross-entropy", enum_values=["cross-entropy", "triplet"],
                      help="loss_type  either cross-entropy  or triplet.")
    flags.DEFINE_float('lr', help="learning_rate", default=1e-3)
    flags.DEFINE_integer('e', help="number of epochs", default=50)
    flags.DEFINE_float('margin', help="margin for triplet loss", default=1.0)
    flags.DEFINE_enum(name="lbl",  default="lda", enum_values=["lda", "knn"],
                      help="Specify labelling method either LDA or KNN.")
    flags.DEFINE_boolean('sw', help="save weights", default=False)
    flags.DEFINE_string('g', help="gpu id", default="0")


def save_weights(model, conv_base,  _accuracy):
    if FLAGS.sw:  # if save weights, will be saved to ./weights/{network}-{dataset}-{loss-type}-{epochs}-{acc}-*.h5
        os.makedirs("./weights/", exist_ok=True)  # create directory if not exists
        save_str = "./weights/{}-{}-{}-epochs-{}-acc-{:.2f}".format(FLAGS.n, FLAGS.d.replace("/","_"), FLAGS.lt,
                                                                    FLAGS.e, _accuracy)
        dump_weights(model, conv_base, save_str, include_top=False)
        print("weights saved as ", save_str)


def main(argv):
    dt1 = datetime.datetime.now()
    del argv  # not used
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.g
    # tf.autograph.set_verbosity(0)
    ds_train, ds_test, size = get_dataset(FLAGS.d, FLAGS.bs)
    # get model
    model, conv_base = get_model(FLAGS.n, size=size, loss_type=FLAGS.lt)
    # setup log directory
    log_name = FLAGS.lt + "-"
    log_dir = 'logs/' + FLAGS.d + '/' + FLAGS.n + '/'
    os.makedirs(log_dir, exist_ok=True)
    logging.get_absl_handler().use_absl_log_file(log_name, log_dir)
    logging.get_absl_handler().setFormatter(None)
    # log flag values and initial accuracy
    logging.info(FLAGS.flag_values_dict())
    init_accuracy = get_accuracy(model, ds_train, ds_test, loss_type=FLAGS.lt)
    logging.info("=======init  Validation accuracy=========  {:.2f} %".format(init_accuracy))

    def ctrl_c_accuracy():
        accuracy = get_accuracy(model, ds_train, ds_test, loss_type=FLAGS.lt)
        logging.info("=======ctrl_c  Validation accuracy=========  {:.2f} %".format(accuracy))
        save_weights(model, conv_base, accuracy)
        print(program_duration(dt1, 'Killed after Time'))

    def exit_gracefully(signum, frame):
        original_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, original_sigint)
        try:
            if input("\nReally quit? (y/n)> ").lower().startswith('y'):
                ctrl_c_accuracy()
                sys.exit(1)
        except KeyboardInterrupt:
            print("Ok ok, quitting")
            sys.exit(1)

    signal.signal(signal.SIGINT, exit_gracefully)
    # start training
    start_training(model, ds_train, epochs=FLAGS.e, batch_size=FLAGS.bs)
    _accuracy = get_accuracy(model, ds_train, ds_test, loss_type=FLAGS.lt)
    logging.info("=======after training  Validation accuracy=========  {:.2f} %".format(_accuracy))
    save_weights(model, conv_base, _accuracy)
    print(program_duration(dt1, 'Total Time taken '))


if __name__ == '__main__':
    setup_flags()
    FLAGS.alsologtostderr = True  # also show logging info to std output
    app.run(main)
