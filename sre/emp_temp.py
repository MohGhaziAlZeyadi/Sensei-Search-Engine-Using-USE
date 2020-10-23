import tensorflow_hub as hub
import tensorflow as tf

print(tf.__version__ )

embed = hub.load("https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1")
embeddings = embed(["cat is on the mat", "dog is in the fog"])