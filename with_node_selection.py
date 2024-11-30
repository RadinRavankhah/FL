from main import federated_learning
import os

# Disable TensorFlow OneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

n = 3
node_selection = [1, 1, 1]

federated_learning(n,node_selection)