import tensorflow as tf
import collections
import random
import re
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import time
import json
from glob import glob
from PIL import Image
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

# global variables

# I put it outside the function because this needs to be referenced in training. Haozhe 11/09/20
from bleu import BleuScorer

FEATURE_SIZE = 2048
ATTENTION_SIZE = 64
BATCH_SIZE = 64
BUFFER_SIZE = 1000
MAX_LENGTH = 0  # will be updated after pre-processing
IMAGE_EXAMPLE_SIZE = 6000  # only uses 6000 image. using full image set requires too much storage space.
top_k = 5000
image_features_extract_model = None  # will be updated after pre-processing
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
VOCAB_SIZE = top_k + 1
CHECKPOINT_PATH = "./topdown_checkpoints/train"


class EncoderModel(tf.keras.Model):
    def __init__(self, image_embedding_dim):
        super(EncoderModel, self).__init__()

        # encoder layers
        self.encoderFC = tf.keras.layers.Dense(image_embedding_dim)

    '''
    Input: 
        image_feature_vector: image features extracted from the last pretrained CNN layer, (batch_size, 64)
    Output: 
        embedding: encoded vector of image features, which will be passed into the sequence model in the decoder (LSTM)
    '''

    def call(self, image_feature_vector):
        x = self.encoderFC(image_feature_vector)
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
        self.embeddingLayer = tf.keras.layers.Embedding(
            vocab_size, pred_embedding_dim)

        self.decoderLSTM1 = tf.keras.layers.LSTM(hidden_dim, return_state=True)
        self.decoderLSTM2 = tf.keras.layers.LSTM(hidden_dim, return_state=True)

        self.decoderFC = tf.keras.layers.Dense(vocab_size)

    '''
       Input: 
           image_embedding: the output from the encoder. This does not vary with time steps
           previous_predict: the predicted token from the decoder at the time step t-1
           previous_hidden_1: the hidden state from LSTM1 at the time step t-1
           previous_hidden_2: the hidden state from LSTM2 at the time step t-1
       Output: 
           predict: the predicted token for the time step t
           hidden_state: the hidden state from the decoder LSTM at the time step t
       '''

    def call(self, image_embedding, previous_predict, previous_hidden_2):
        # A. top-down attention LSTM:
        # use image_embedding and previous_hidden_state to produce attention weights
        # use this attention weights to filter important features from image_embedding

        prev_hidden_2_embedding = tf.expand_dims(previous_hidden_2, 1)
        prev_token_embedding = self.embeddingLayer(previous_predict)
        input_to_lstm_1 = tf.concat([prev_hidden_2_embedding, image_embedding, prev_token_embedding], axis=1)
        _, hidden_1, _ = self.decoderLSTM1(input_to_lstm_1)

        at_image = self.attendImage(image_embedding)
        at_pred = self.attendHidden(hidden_1)
        score = self.attendDense(tf.nn.tanh(at_image + at_pred))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * image_embedding
        # this is the attended image vector
        context_vector = tf.reduce_sum(context_vector, axis=1)

        # print("context vector")
        # print(context_vector.shape)

        # B. decoding from attended context vector:

        # concatenate context_vector and predict_embedding to use as the input to LSTM
        hidden_1_embedding = tf.expand_dims(hidden_1, 1)
        input_to_lstm_2 = tf.concat(
            [tf.reshape(tf.expand_dims(context_vector, 1), [image_embedding.shape[0], 1, -1]), hidden_1_embedding],
            axis=1)
        _, hidden_2, _ = self.decoderLSTM1(input_to_lstm_2)

        predict = self.decoderFC(hidden_2)

        return predict, hidden_2, attention_weights


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
    img_tensor = np.load(img_name.decode('utf-8') + '.npy')
    return img_tensor, cap


'''
Note added by Haozhe 11/27/20:
The dataset is extremely large (15 gb for zip files, and more after extracted)
My own PC and Purdue CS CUDA machine does not have enough storage quota for the dataset
I upgraded my google drive to 100 gb and downloaded it to google drive and run it with Colab
For some reason, using this code to download the image dataset seems to give a corrupted train2014.zip
I ended up using wget to download and then use 7za to unzip it before running this program
'''


def getRawData():
    print("---getRawData---")
    annotation_folder = './annotations/'
    annotation_file = './annotations/captions_train2014.json'
    '''if not os.path.exists(os.path.abspath('.') + annotation_folder):
        annotation_zip = tf.keras.utils.get_file('captions.zip',
                                                 cache_subdir=os.path.abspath(
                                                     '.'),
                                                 origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
                                                 extract=True)
        annotation_file = os.path.dirname(
            annotation_zip) + '/annotations/captions_train2014.json'
        os.remove(annotation_zip)'''

    # Download image files
    image_folder = '/content/drive/MyDrive/train2014/'
    ''' if not os.path.exists(os.path.abspath('.') + image_folder):
        image_zip = tf.keras.utils.get_file('train2014.zip',
                                            cache_subdir=os.path.abspath('.'),
                                            origin='http://images.cocodataset.org/zips/train2014.zip',
                                            extract=True)
        PATH = os.path.dirname(image_zip) + image_folder
        os.remove(image_zip)
    else:'''

    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    # Group all captions together having the same image ID.
    image_path_to_caption = collections.defaultdict(list)
    for val in annotations['annotations']:
        caption = f"<start> {val['caption']} <end>"
        image_path = '/content/drive/MyDrive/train2014/COCO_train2014_' + '%012d.jpg' % (val['image_id'])
        image_path_to_caption[image_path].append(caption)
    # image_paths = list(image_path_to_caption.keys())
    # use images that exist in the files, rather than those
    image_paths = [join(image_folder, f) for f in listdir(image_folder) if isfile(join(image_folder, f))]
    image_paths = image_paths[:IMAGE_EXAMPLE_SIZE]
    print("image_paths: " + str(len(image_paths)) + str(image_paths[:5]))
    random.shuffle(image_paths)
    return image_paths, image_path_to_caption


def preprocessData(raw):
    print("---preprocessData---")
    global MAX_LENGTH
    global image_features_extract_model

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

    image_dataset = tf.data.Dataset.from_tensor_slices(image_paths)
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
    # Note: only need to run in the frist time. Haozhe 11/25/20
    '''for img, path in tqdm(image_dataset):
        batch_features = image_features_extract_model(img)
        batch_features = tf.reshape(batch_features,
                                    (batch_features.shape[0], -1, batch_features.shape[3]))
        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            np.save(path_of_feature, bf.numpy())'''

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
    print("padding: ")
    print(train_seqs[:5])
    # Pad each vector to the max_length of the captions
    # If you do not provide a max_length value, pad_sequences calculates it automatically
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(
        train_seqs, padding='post')

    # Calculates the max_length, which is used to store the attention weights
    MAX_LENGTH = calc_max_length(train_seqs)

    # Split the data into training and testing

    img_to_cap_vector = collections.defaultdict(list)
    for img, cap in zip(img_name_vector, cap_vector):
        img_to_cap_vector[img].append(cap)

    # Create training and validation sets using an 80-20 split randomly.
    img_keys = list(img_to_cap_vector.keys())
    random.shuffle(img_keys)

    slice_index = int(len(img_keys) * 0.8)
    img_name_train_keys, img_name_val_keys = img_keys[:
                                                      slice_index], img_keys[slice_index:]
    print("parsing dataset")

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
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset, (img_name_val, cap_val)


def trainModel(dataset, epochs=5000):
    print("---trainModel---")
    encoder = EncoderModel(image_embedding_dim=128)
    decoder = DecoderModel(pred_embedding_dim=128,
                           hidden_dim=128, vocab_size=VOCAB_SIZE)

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    # Create checkpoint manager (and read from checkpoint dir)
    ckpt = tf.train.Checkpoint(encoder=encoder,
                               decoder=decoder,
                               optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(
        ckpt, CHECKPOINT_PATH, max_to_keep=5)

    # Restore checkpoint if needed
    start_epoch = 0
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        ckpt.restore(ckpt_manager.latest_checkpoint)

    def loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_mean(loss_)

    # Continue from previous epoch if needed
    for epoch in range(start_epoch, epochs):
        total_loss = 0
        for (batch, (image, caption)) in enumerate(dataset):
            if image.shape[0] != BATCH_SIZE:
                # print(image.shape)
                continue
            loss = 0
            # use 0s as the initial hidden state to feed in the LSTM
            previous_hidden_state = tf.zeros((BATCH_SIZE, decoder.hidden_dim))
            # use <start> as the initial predict to feed in the LSTM
            previous_decoder_predict = tf.expand_dims(
                [tokenizer.word_index['<start>']] * caption.shape[0], 1)

            with tf.GradientTape() as tape:
                image_features = encoder.call(image)

                # teacher forcing to predict each token in the caption
                for i in range(1, caption.shape[1]):
                    # passing the features through the decoder
                    predict, hidden, _ = decoder.call(image_embedding=image_features,
                                                      previous_predict=previous_decoder_predict,
                                                      previous_hidden_2=previous_hidden_state)
                    loss += loss_function(caption[:, i], predict)

                    # needs to expand the dimension to match the input shape of the decoder
                    previous_decoder_predict = tf.expand_dims(caption[:, i], 1)
                    previous_hidden_state = hidden

            trainable_variables = encoder.trainable_variables + decoder.trainable_variables
            gradients = tape.gradient(loss, trainable_variables)
            optimizer.apply_gradients(zip(gradients, trainable_variables))

            # average over the length of caption
            total_loss = total_loss + (loss / int(caption.shape[1]))

        if epochs % 100 == 0:
            print('Epoch {} Loss {:.4f}'.format(
                epoch + 1, total_loss))
            # Checkpoint saving
            ckpt_manager.save()

    return encoder, decoder


def evaluate(image, encoder, decoder):
    print("---evaluate---")

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, [1, ATTENTION_SIZE, -1])
    # print(img_tensor_val.shape)

    features = encoder(img_tensor_val)
    print(features.shape)
    # use 0s as the initial hidden state to feed in the LSTM
    previous_hidden_state = tf.zeros((1, decoder.hidden_dim))

    previous_decoder_predict = tf.expand_dims([tokenizer.word_index['<start>']], 0)

    result = ['<start>']

    for i in range(MAX_LENGTH):
        previous_decoder_predict, previous_hidden_state, attention_weights = decoder(
            features, previous_decoder_predict, previous_hidden_state)

        predicted_id = tf.random.categorical(previous_decoder_predict, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result

        previous_decoder_predict = tf.expand_dims([predicted_id], 0)

    return result

# Run captioning on validation set
# This only runs on 1 random sample in our dataset.
# This is for previewing the results only, since we will have a separate evaluation script
def runModel(model, test_set):
    test_image, test_caption = test_set
    print("---runModel---")
    encoder, decoder = model
    references = []
    candidates = []
    for idx in range(0, len(test_image)):
        image = test_image[idx]
        print("Test Image: " + str(image))
        real_caption = ' '.join([tokenizer.index_word[i]
                                 for i in test_caption[idx] if i not in [0]])
        result = evaluate(image, encoder, decoder)
        pred_caption = ' '.join(result)
        print('Real Caption:', real_caption)
        print('Prediction Caption:', pred_caption)
        references.append(real_caption)
        candidates.append(pred_caption)

    scorer = BleuScorer(references, candidates)
    bleu1, bleu4 = scorer.compute_score()
    print('BLUE-1: ', bleu1)
    print('BLUE-4: ', bleu4)


def main():
    print("Image Captioning Main")
    raw = getRawData()
    dataset_train, dataset_test = preprocessData(raw)
    model = trainModel(dataset_train)
    runModel(model, dataset_test)


if __name__ == '__main__':
    main()
