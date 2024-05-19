# TensorFlow provides a built-in function to check its version and built-in compiler information.
import tensorflow as tf

# Print TensorFlow version
print("TensorFlow version:", tf.__version__)

# Print CUDA and cuDNN version if available
print("Built with CUDA:", tf.test.is_built_with_cuda())

# Check if a GPU is available and its details
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        print(f"GPU Name: {gpu.name}, Type: {gpu.device_type}")
else:
    print("No GPUs found.")

# TensorFlow does not directly provide the CUDA and cuDNN versions through its API.
# However, it does provide the details about which version it was compiled against.
# We'll print the C++ library versions used by TensorFlow which usually includes the CUDA and cuDNN versions.
print("TensorFlow was built with the following C++ library versions:")
print("CUDA version:", tf.sysconfig.get_build_info()["cuda_version"])
print("cuDNN version:", tf.sysconfig.get_build_info()["cudnn_version"])
