import tensorflow as tf
import collections
import random
import re
import numpy as np
import os
import time
import json
from glob import glob
from PIL import Image
import pickle


# global variables

# I put it outside the function because this needs to be referenced in training. Haozhe 11/09/20

BATCH_SIZE = 64
BUFFER_SIZE = 1000
top_k = 5000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
VOCAB_SIZE = top_k+1



class EncoderModel(tf.keras.Model):
    def __init__(self, image_embedding_dim):
        super(EncoderModel, self).__init__()

        # encoder layers
        self.encoderFC = tf.keras.Dense(image_embedding_dim)

    '''
    Input: 
        image_feature_vector: image features extracted from the last pretrained CNN layer, (batch_size, 64)
    Output: 
        embedding: encoded vector of image features, which will be passed into the sequence model in the decoder (LSTM)
    '''
    def call(self, image_feature_vector):
        x = self.fc(image_feature_vector)
        x = tf.nn.relu(x)
        return x


class DecoderModel(tf.keras.Model):
    def __init__(self, pred_embedding_dim, hidden_dim, vocab_size):
        super(DecoderModel, self).__init__()

        self.pred_embedding_dim = pred_embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        # attention layers
        self.attendImage = tf.keras.layers.Dense(hidden_dim)
        self.attendHidden = tf.keras.layers.Dense(hidden_dim)
        self.attendDense = tf.keras.layers.Dense(1)

        # decoder layers
        self.embeddingLayer = tf.keras.Embedding(vocab_size, pred_embedding_dim)
        self.decoderGRU = tf.keras.layers.GRU(hidden_dim)
        self.decoderFC1 = tf.keras.Dense(hidden_dim)
        self.decoderFC2 = tf.keras.Dense(vocab_size)

    '''
       Input: 
           image_embedding: the output from the encoder. This does not vary with time steps
           previous_predict: the predicted token from the decoder at the time step t-1
           previous_hidden_state: the hidden state from the decoder GRU at the time step t-1
       Output: 
           predict: the predicted token for the time step t
           hidden_state: the hidden state from the decoder GRU at the time step t
       '''
    def call(self, image_embedding, previous_predict, previous_hidden_state):

        # A. attention mechanism:
        # uses image_embedding and previous_hidden_state to produce a weighted context vector

        hidden_with_time_axis = tf.expand_dims(previous_hidden_state, 1)
        at_image = self.attendImage(image_embedding)
        at_pred = self.attendHidden(previous_hidden_state)
        context_vector = self.attendDense(tf.nn.tanh(at_image + at_pred))

        # B. decoding from attended context vector:

        # map the previous prediction (a word token) into embedding (a vector representation of that word)
        # so that it can be feed into the GRU layer
        predict_embedding = self.embeddingLayer(previous_predict)

        # concatenate context_vector and predict_embedding to use as the input to GRU
        # concat_embedding: (batch_size, 1, pred_embedding_dim + hidden_size)
        concat_embedding = tf.concat([tf.expand_dims(context_vector, 1), predict_embedding], axis=-1)

        # passing the concatenated vector to the GRU
        predict, hidden_state = self.decoderGRU(concat_embedding)

        # shape == (batch_size, max_length, hidden_size)
        predict = self.decoderFC1(predict)
        # x shape == (batch_size * max_length, hidden_size)
        predict = tf.reshape(predict, (-1, predict.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        predict = self.decoderFC2(predict)

        return predict, hidden_state


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


def calc_max_length(tensor):
    return max(len(t) for t in tensor)

# Load the numpy files


def map_func(img_name, cap):
    img_tensor = np.load(img_name.decode('utf-8')+'.npy')
    return img_tensor, cap


def getRawData():
    annotation_folder = '/annotations/'
    annotation_file = ''
    if not os.path.exists(os.path.abspath('.') + annotation_folder):
        annotation_zip = tf.keras.utils.get_file('captions.zip',
                                                 cache_subdir=os.path.abspath(
                                                     '.'),
                                                 origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                                 extract=True)
        annotation_file = os.path.dirname(
            annotation_zip) + '/annotations/captions_train2014.json'
        os.remove(annotation_zip)

    # Download image files
    image_folder = '/train2014/'
    if not os.path.exists(os.path.abspath('.') + image_folder):
        image_zip = tf.keras.utils.get_file('train2014.zip',
                                            cache_subdir=os.path.abspath('.'),
                                            origin='http://images.cocodataset.org/zips/train2014.zip',
                                            extract=True)
        PATH = os.path.dirname(image_zip) + image_folder
        os.remove(image_zip)
    else:
        PATH = os.path.abspath('.') + image_folder

    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    # Group all captions together having the same image ID.
    image_path_to_caption = collections.defaultdict(list)
    for val in annotations['annotations']:
        caption = f"<start> {val['caption']} <end>"
        image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (val['image_id'])
        image_path_to_caption[image_path].append(caption)
    image_paths = list(image_path_to_caption.keys())
    random.shuffle(image_paths)
    return image_paths, image_path_to_caption


def preprocessData(raw):
    image_paths, image_path_to_caption = raw

    # before pre-processing, each image is corresponding to multiple caption.
    # we will duplicate the images so that we have (image, caption) pairs

    train_captions = []
    img_name_vector = []
    for image_path in image_paths:
        caption_list = image_path_to_caption[image_path]
        train_captions.extend(caption_list)
        img_name_vector.extend([image_path] * len(caption_list))
    encode_train = sorted(set(img_name_vector))

    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    image_dataset = image_dataset.map(
        load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

    # pretrained InceptionV3 to extract features from images
    image_model = tf.keras.applications.InceptionV3(
        include_top=False, weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    # cache the features of image extracted by InceptionV3 to the disk
    # because the memory in RAM is not sufficient to store these features for all images
    for img, path in image_dataset:
        batch_features = image_features_extract_model(img)
        batch_features = tf.reshape(batch_features,
                                    (batch_features.shape[0], -1, batch_features.shape[3]))
        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            np.save(path_of_feature, bf.numpy())

    # img_name_vector is a list of image file paths
    # train_captions is a list of corresponding captions
    # we need to split the training and testing set from this
    # return img_name_vector, train_captions

    # Preprocess and tokenize the captions

    # Choose the top 5000 words from the vocabulary

    tokenizer.fit_on_texts(train_captions)
    train_seqs = tokenizer.texts_to_sequences(train_captions)

    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'

    # Create the tokenized vectors
    train_seqs = tokenizer.texts_to_sequences(train_captions)

    # Pad each vector to the max_length of the captions
    # If you do not provide a max_length value, pad_sequences calculates it automatically
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(
        train_seqs, padding='post')

    # Calculates the max_length, which is used to store the attention weights
    max_length = calc_max_length(train_seqs)

    # Split the data into training and testing

    img_to_cap_vector = collections.defaultdict(list)
    for img, cap in zip(img_name_vector, cap_vector):
        img_to_cap_vector[img].append(cap)

    # Create training and validation sets using an 80-20 split randomly.
    img_keys = list(img_to_cap_vector.keys())
    random.shuffle(img_keys)

    slice_index = int(len(img_keys)*0.8)
    img_name_train_keys, img_name_val_keys = img_keys[:
                                                      slice_index], img_keys[slice_index:]

    img_name_train = []
    cap_train = []
    for imgt in img_name_train_keys:
        capt_len = len(img_to_cap_vector[imgt])
        img_name_train.extend([imgt] * capt_len)
        cap_train.extend(img_to_cap_vector[imgt])

    img_name_val = []
    cap_val = []
    for imgv in img_name_val_keys:
        capv_len = len(img_to_cap_vector[imgv])
        img_name_val.extend([imgv] * capv_len)
        cap_val.extend(img_to_cap_vector[imgv])

    # Create a tf.data dataset for training
    num_steps = len(img_name_train) // BATCH_SIZE
    # Shape of the vector extracted from InceptionV3 is (64, 2048)
    # These two variables represent that vector shape
    features_shape = 2048
    attention_features_shape = 64

    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

    # Use map to load the numpy files in parallel
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(
        map_func, [item1, item2], [tf.float32, tf.int32]),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle and batch
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset


def trainModel(dataset, epochs=10000):
    encoder = EncoderModel(image_embedding_dim=128)
    decoder = DecoderModel(pred_embedding_dim=128, hidden_dim=128, vocab_size=VOCAB_SIZE)

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    for epoch in range(epochs):
        total_loss = 0
        for (batch, (image, caption)) in enumerate(dataset):
            loss = 0
            # use 0s as the initial hidden state to feed in the GRU
            previous_hidden_state = tf.zeros((BATCH_SIZE, decoder.hidden_dim))
            # use <start> as the initial predict to feed in the GRU
            previous_decoder_predict = tf.expand_dims([tokenizer.word_index['<start>']] * caption.shape[0], 1)

            with tf.GradientTape() as tape:
                image_features = encoder.call(image)

                # teacher forcing to predict each token in the caption
                for i in range(1, caption.shape[1]):
                    # passing the features through the decoder
                    predict, hidden = decoder.call(image_embedding=image_features,
                                                   previous_predict=previous_decoder_predict,
                                                   previous_hidden_state=previous_hidden_state)
                    loss += loss_function(caption[:, i], predict)

                    # needs to expand the dimension to match the input shape of the decoder
                    previous_decoder_predict = tf.expand_dims(caption[:, i], 1)

            trainable_variables = encoder.trainable_variables + decoder.trainable_variables
            gradients = tape.gradient(loss, trainable_variables)
            optimizer.apply_gradients(zip(gradients, trainable_variables))

            total_loss = total_loss + (loss / int(caption.shape[1]))  # average over the length of caption

        if epochs % 100 == 0:
            print('Epoch {} Loss {:.4f}'.format(
                epoch + 1, total_loss))
        # TODO: checkpoint saving
        '''
        if epoch % 5 == 0:
            ckpt_manager.save()

        '''

        return encoder, decoder


def runModel(test_x, test_y):
    # TODO: complete the model
    pass


def main():
    print("Image Captioning Main")
    raw = getRawData()
    dataset = preprocessData(raw)
    model = trainModel(dataset)
    '''preds = runModel(data_, model)
    evalResults(data[1], preds)

    # classification on IRIS
    print("Classification on IRIS")
    raw = getRawDataIris()
    data = preprocessDataIris(raw)
    model = trainModelIris(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)'''


if __name__ == '__main__':
    main()
