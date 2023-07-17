# Cancer Type Prediction Using Transfer Learning

This project focuses on predicting the type of cancer based on the Medical Text Dataset - Cancer Doc Classification. The dataset consists of medical texts categorized into three types of cancer: colon cancer, lung cancer, and thyroid cancer. The number of samples in each category is as follows: colon cancer=2579, lung cancer=2180, thyroid cancer=2810.

## Project Overview

The project employs transfer learning techniques using TensorFlow Hub, which provides pre-trained machine learning models ready for fine-tuning and deployment. Two models, namely Universal Sentence Encoder and NNLM (Neural Network Language Model), are selected for fine-tuning on the dataset.

### Data Collection

The data was collected using the Kaggle API extended, which facilitated loading the data directly from Kaggle. The Pandas library's `read_csv` function was utilized to handle unzipping the compressed data and formatting it into a dataframe.

### Model Architecture

TensorFlow Hub is used for the preprocessing of text data and feature extraction. The selected pre-trained models, which utilize context-based embeddings, are loaded using a single line of code:

```python
hub_layer = hub.KerasLayer(module_url, output_shape=[embed_size], input_shape=[], dtype=tf.string, trainable=False)
```

The `hub_layer` is added to the sequential layer of the model and further connected to Dense layers in the neural network.

### Callbacks and Visualization

To enhance training and monitor the model's performance, several callbacks are employed:

- `EpochDots`: A module from the TensorFlow Docs library to conceal all training information during the execution of the TensorFlow Keras model.
- `EarlyStopping`: A callback from the `tf.keras.callbacks` module to stop training when the monitored metric has stopped improving.
- `TensorBoard`: A callback from the TensorFlow callbacks module to log the model's information to a specified log directory.

For visualization purposes, the `HistorPlotter` module from `tensorflow_docs.plots` is used to plot the history information, including accuracy and loss, for both the Universal Sentence Encoder and NNLM models. Additionally, the `TensorBoard` library is employed to monitor training metrics from the training log information stored in the log directory.

## Conclusion

By leveraging transfer learning and employing pre-trained models, this project successfully predicts the type of cancer based on medical text data. The combination of TensorFlow Hub, TensorFlow callbacks, and visualization techniques provides an effective solution for cancer type classification.
