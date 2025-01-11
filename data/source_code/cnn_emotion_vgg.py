def _build_train_model(self):
    # Define input shape for grayscale images
    input_shape = (self._img_dim[0], self._img_dim[1], 1)
    
    # Create input layer and convert to 3 channels
    inputs = keras.Input(shape=input_shape)
    x = keras.layers.Concatenate()([inputs, inputs, inputs])  # Convert to 3 channels
    
    # Create VGG16 base model
    base_model = keras.applications.VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(self._img_dim[0], self._img_dim[1], 3),
        pooling='avg'
    )
    
    # Add the VGG16 base 
    x = base_model(x)
    
    # Add custom classification layers
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(256, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(len(self._g_emotion_labels), activation='softmax')(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
    #raise NotImplementedError('1X: Keras model architecture needs to be implemented') 
