import os

SHOW_TF_WARNINGS = False
if not SHOW_TF_WARNINGS:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Avoid annoying tf warnings

try:
    import tensorflow as tf

    GPU_LIST = tf.config.experimental.list_physical_devices("GPU")
    if GPU_LIST:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in GPU_LIST:
                # Prevent Tensorflow to take all GPU memory
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices("GPU")
                print(
                    len(GPU_LIST), "Physical GPUs,", len(logical_gpus), "Logical GPUs"
                )
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    TF_LOADED = True
except:
    TF_LOADED = False

TF_STATUS_STR = (
    (
        "TensorFlow running ({} GPU)". format(len(GPU_LIST))
    )
    if TF_LOADED
    else "TensorFlow not found."
)