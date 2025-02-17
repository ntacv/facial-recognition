# -*- coding: utf-8 -*-
"""
Created on Tue Dec 29 16:59:16 2020

@author: pbourdon
"""

import os, glob, re, sys, argparse, itertools
import keras.optimizers
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import sklearn, sklearn.preprocessing, sklearn.model_selection, sklearn.metrics
import keras, keras.backend, keras.preprocessing.image, keras.utils.np_utils
import keras.layers as layers
#from keras.preprocessing.image import ImageDataGenerator

class KerasFaceAnalyzerBase():
    _g_ck_emotion_dict = {None:'???', 0:'Neutral', 1:'Anger', 2:'???', 3:'Disgust', 4:'Fear', 5:'Happiness', 6:'Sadness', 7:'Surprise'}
    _g_emotion_labels = ['Neutral','Anger','Disgust','Fear','Happiness','Sadness','Surprise']
    _g_emotion_map = {e:idx for idx, e in enumerate(_g_emotion_labels)}
    _g_fname_ext_model_weights = '.weights.h5'
    _g_fname_ext_model_history = '.history.csv'
    _g_fname_ext_model_per_epoch = '.epoch{epoch:02d}-loss{val_loss:.2f}.h5'
    _g_fname_ext_model = '.json'

    def __init__(self):
        self._img_dim = (64,64)
        self._data_generator_preproc = None
        self._X_train_fname = 'X_train.npy'
        self._Y_train_fname = 'Y_train.npy'
        self._X_test_fname = 'X_test.npy'
        self._Y_test_fname = 'Y_test.npy'

    @classmethod
    def _load_ck_sample(cls, path):
        basename = os.path.basename(path)
        r = re.compile('S(\d+)_(\d+)_(\d+)_landmarks.txt')
        res = r.findall(basename)
        assert len(res)==1, 'File does not match Cohn-Kanade pattern'
        subject_id, session_id, _ = res[0]
        subject_id = int(subject_id)
        session_id = int(session_id)
        name = basename.replace('_landmarks.txt', '')
    
        assert os.path.exists(path), 'Landmarks file {} does not exist'.format(path)
        img_path = path.replace('_landmarks.txt', '.png')
        assert os.path.exists(img_path), 'Image file {} does not exist'.format(img_path)
        img = cv2.imread(img_path, flags=cv2.IMREAD_GRAYSCALE).astype(float)/255.0

        emotion_path = path.replace('_landmarks.txt', '_emotion.txt')
        assert os.path.exists(emotion_path), 'Emotion label file {} does not exist'.format(emotion_path)
    
        with open(path, 'r') as f:
            landmarks = np.array([[float(x) for x in line.split()] for line in f])
        assert landmarks.shape[0]!=0, 'No landmark found in file {}'.format(path)
    
        with open(emotion_path, 'r') as f:
            emotion_label = cls._g_ck_emotion_dict.get(float(f.readline()))
            emotion_index = cls._g_emotion_map.get(emotion_label)
            
        return img, landmarks, emotion_index, emotion_label

    def _load_ck_data(self, input_dir):
        search_pattern = os.path.join(input_dir, 'S[0-9][0-9][0-9]_[0-9][0-9][0-9]_[0-9][0-9][0-9][0-9][0-9][0-9][0-9][0-9]_landmarks.txt')
        files = glob.glob(search_pattern)
        assert len(files)>0, 'No file found in input directory'
        # print('Found {} files'.format(len(files)))

        ck_data_dict = {'img':[], 'emotion_index':[], 'emotion_label':[]}
    
        n_landmarks = None
        for path in files:
            img, landmarks, emotion_index, emotion_label = self._load_ck_sample(path)
            if emotion_label not in self._g_emotion_labels: continue
            
            x_min, x_max = int(np.min(landmarks[:,0])), int(np.max(landmarks[:,0]))
            y_min, y_max = int(np.min(landmarks[:,1])), int(np.max(landmarks[:,1]))
            img_crop = img[y_min:y_max, x_min:x_max]
            img_crop = cv2.resize(img_crop, self._img_dim)

            if n_landmarks is None: n_landmarks = landmarks.shape[0]
            assert landmarks.shape[0]==n_landmarks, 'Mismatch in number of landmarks'

            if keras.backend.image_data_format() == 'channels_last':
                img_crop = img_crop.reshape(img_crop.shape[0], img_crop.shape[1], 1)    # tensorflow
            elif keras.backend.image_data_format() == 'channels_first':
                img_crop = img_crop.reshape(1, img_crop.shape[0], img_crop.shape[1])    # theano
            ck_data_dict['img'].append(img_crop)
            ck_data_dict['emotion_index'].append(emotion_index)
            ck_data_dict['emotion_label'].append(emotion_label)
            
        return ck_data_dict
    
    def prepare_data(self, input_dir, output_dir):
        print('Loading data...')
        ck_data_df = pd.DataFrame.from_dict(self._load_ck_data(input_dir))
        print('Found {} samples'.format(len(ck_data_df)))
        print(ck_data_df['emotion_label'].value_counts().apply(lambda x: '{:.2f}%'.format(x/len(ck_data_df)*100)))

        print('Normalizing...')
        # create min max normalized column
        X = np.asarray(ck_data_df['img'].values.tolist())
        X = X.reshape(len(X), -1)
        min_max_scaler = sklearn.preprocessing.MinMaxScaler()
        X_min_max = min_max_scaler.fit_transform(X)
        if keras.backend.image_data_format() == 'channels_last':
            ck_data_df['normalized_img'] = [arr.reshape(self._img_dim[0], self._img_dim[1], 1) for arr in X_min_max]    # tensorflow
        elif keras.backend.image_data_format() == 'channels_first':
            ck_data_df['normalized_img'] = [arr.reshape(1, self._img_dim[0], self._img_dim[1]) for arr in X_min_max]    # theano

        print('Splitting for cross-validation...')
        X = np.array(ck_data_df['normalized_img'].values.tolist())
        y = np.array(ck_data_df['emotion_index'].values.tolist())
        assert len(X)==len(y), 'Groundtruth error'
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=.3, random_state=42)
        print('Training Set: {} samples'.format(len(X_train)))
        print('Test Set: {} samples'.format(len(X_test)))

        print('Saving Keras-compliant data (one-hot encoded)...')
        Y_train = keras.utils.np_utils.to_categorical(y_train)
        Y_test = keras.utils.np_utils.to_categorical(y_test)
        print('X:', X_train.shape, X_test.shape)
        print('Y:', Y_train.shape, Y_test.shape)

        for arr, fn in ((X_train, self._X_train_fname), (Y_train, self._Y_train_fname), (X_test, self._X_test_fname), (Y_test, self._Y_test_fname)):
            path = os.path.join(output_dir, fn)
            print('{}'.format(path))
            np.save(path, arr)

    def _load_training_data(self, output_dir):
        print('Loading training data...')
        X_train = np.load(os.path.join(output_dir, self._X_train_fname))
        Y_train = np.load(os.path.join(output_dir, self._Y_train_fname))
        return X_train, Y_train

    def _load_test_data(self, output_dir):
        print('Loading test data...')
        X_test = np.load(os.path.join(output_dir, self._X_test_fname))
        Y_test = np.load(os.path.join(output_dir, self._Y_test_fname))
        return X_test, Y_test

    def _build_train_model(self):
        
        input_shape = (self._img_dim[0], self._img_dim[1], 1)
        
        """ if keras.backend.image_data_format() == 'channels_last':
            input_shape = (self._img_dim[0], self._img_dim[1], 1)
        elif keras.backend.image_data_format() == 'channels_first':
            input_shape = (1, self._img_dim[0], self._img_dim[1])
 """
        model = keras.models.Sequential([
            layers.Conv2D(32, (3, 3), padding='same', input_shape=input_shape),
            layers.Activation('relu'),
            layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.2),
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.Flatten(),
            layers.Dense(1024),
            layers.Flatten(),
            layers.Dense(512),
            layers.Activation('relu'),
            layers.Dropout(0.2),
            layers.Dense(7, activation='softmax')
        ])

        return model
        #raise NotImplementedError('1X: Keras model architecture needs to be implemented') 

    def _compile_train_model(self, model):
        
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])
        
        #raise NotImplementedError('2X: Keras model compilation needs to be implemented') 
    
    def _build_data_generator(self, output_dir):
        X_train, Y_train = self._load_training_data(output_dir)

        print('Building image data generator...')
        # keras.backend.set_image_data_format('channels_last')
        data_generator = keras.preprocessing.image.ImageDataGenerator(rescale=1./255, rotation_range=10,
                                                                      shear_range=0.2, width_shift_range=0.2,
                                                                      height_shift_range=0.2, horizontal_flip=True,
                                                                      preprocessing_function=self._data_generator_preproc)
        data_generator.fit(X_train)
        print(data_generator.flow(X_train, Y_train, batch_size=50000).next()[0].shape)
        
        return data_generator

    def train(self, output_dir, model_fname, epochs=5):
        X_train, Y_train = self._load_training_data(output_dir)
        balance = [np.sum(Y_train[:,i])/Y_train.shape[0] for i in range(len(self._g_emotion_labels))]
        for e, b in zip(self._g_emotion_labels, balance):
            print('{}: {:.2f}%'.format(e, 100*b))

        data_generator = self._build_data_generator(output_dir)

        print('Building model...')
        model = self._build_train_model()
        self._compile_train_model(model)
        print(model.summary())

        print('Training...')
        model_path = os.path.join(output_dir, model_fname)
        history_path = model_path+self._g_fname_ext_model_history
        model_cb = keras.callbacks.ModelCheckpoint(filepath=model_path+self._g_fname_ext_model_per_epoch, 
                                                   monitor='val_accuracyh', save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')
        history_cb = keras.callbacks.CSVLogger(history_path, separator=",", append=False)

        history = model.fit(X_train, Y_train, verbose=1, shuffle=True, epochs=epochs, validation_split=.2, callbacks=[model_cb, history_cb])

        print('Saving model...')
        with open(model_path+self._g_fname_ext_model, 'w') as json_file:
            json_file.write(model.to_json())
        model.save_weights(model_path+self._g_fname_ext_model_weights)
        model.save(model_path)

    def _plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title if not normalize else title+' (normalized)')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
    
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            value = cm[i, j] if not normalize else '{:.2f}%'.format(100*cm[i, j])
            plt.text(j, i, value, horizontalalignment='center', color='red' if cm[i, j] > thresh else 'black')
    
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
    def display_results(self, output_dir, model_fname):
        model_path = os.path.join(output_dir, model_fname)
        history_path = model_path+self._g_fname_ext_model_history

        print('Loading model...')
        # with open(model_path+self._g_fname_ext_model, 'r') as json_file:
        #     model = keras.models.model_from_json(json_file.read())
        # model.load_weights(model_path+self._g_fname_ext_model_weights)
        model = keras.models.load_model(model_path)
        self._compile_train_model(model)

        print('Loading training history...')
        """
        history = pd.read_csv(history_path, sep=',', engine='python')

        plt.figure()
        plt.plot(history['epoch'], history['accuracy'])
        plt.plot(history['epoch'], history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

        plt.figure()
        plt.plot(history['epoch'], history['loss'])
        plt.plot(history['epoch'], history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        plt.show()

        X_train, Y_train = self._load_training_data(output_dir)
        print('Evaluating model on train data...')
        score = model.evaluate(X_train, Y_train, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
        """
    def test(self, output_dir, model_fname):
        model_path = os.path.join(output_dir, model_fname)
        history_path = model_path+self._g_fname_ext_model_history

        print('Loading model...')
        model = keras.models.load_model(model_path)
        self._compile_train_model(model)

        X_test, Y_test = self._load_test_data(output_dir)
        print('Evaluating model on test data...')
        score = model.evaluate(X_test, Y_test, verbose=0)
        print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

        print('Computing confusion matrix on test data...')
        Y_test_pred = model.predict(X_test)
        Y_test_pred = np.argmax(Y_test_pred, axis=1)
        Y_test = np.argmax(Y_test, axis=1)
        
        Y_test_pred = [self._g_emotion_labels[y] for y in Y_test_pred]
        Y_test = [self._g_emotion_labels[y] for y in Y_test]
        confusion_matrix = sklearn.metrics.confusion_matrix(Y_test, Y_test_pred, labels=self._g_emotion_labels)
        print('Confusion matrix:\n', confusion_matrix)
        # accuracy = sklearn.metrics.accuracy_score(Y_test, Y_test_pred)
        # print('Accuracy: {:.2f}%'.format(100*accuracy))
        self._plot_confusion_matrix(confusion_matrix, self._g_emotion_labels, normalize=False)
        self._plot_confusion_matrix(confusion_matrix, self._g_emotion_labels, normalize=True)

class MyKerasFaceAnalyzer(KerasFaceAnalyzerBase):
    def _build_train_model(self):
        if keras.backend.image_data_format() == 'channels_last':
            input_shape=(self._img_dim[0], self._img_dim[1], 1)
        elif keras.backend.image_data_format() == 'channels_first':
            input_shape=(1, self._img_dim[0], self._img_dim[1])
        
        model = keras.models.Sequential()
        model.add(keras.layers.Dense(len(self._g_emotion_labels), activation='softmax'))
        
        return model

    def _compile_train_model(self, model):
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

def main(argv):
    plt.close('all')

    parser = argparse.ArgumentParser(description='CNN facial analysis')
    parser.add_argument('input_dir', help='input directory')
    parser.add_argument('-O', '--output_dir', type=str, default=None, help='output directory')
    parser.add_argument('-M', '--model_fname', type=str, default=None, help='model filename')
    parser.add_argument('-E', '--epochs', type=int, default=1, help='number of epochs (default 1)')

    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir is not None else os.path.curdir
    model_fname = os.path.abspath(args.model_fname) if args.model_fname is not None else 'cnn_face_emotion.model'
    epochs = args.epochs

    analyzer = KerasFaceAnalyzerBase() #KerasFaceAnalyzerBase()
    
    analyzer.prepare_data(input_dir, output_dir)
    analyzer.train(output_dir, model_fname, epochs=epochs)
    analyzer.display_results(output_dir, model_fname)
    analyzer.test(output_dir, model_fname)

if __name__ == '__main__':
    main(sys.argv)



    
