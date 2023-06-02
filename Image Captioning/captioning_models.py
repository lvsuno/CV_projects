from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, GRU
from tensorflow.keras.layers import Add


# define the captioning model
def define_model(vocab_size, max_length):
    # feature extractor model (image)
    #inputs1 = Input(shape=(4096,))
    inputs1 = Input(shape=(1280,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    # sequence model (Text)
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = GRU(256, return_sequences=True)(se2)
    se3 = GRU(256)(se3)
    # decoder model
    decoder1 = Add()([fe2,se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # summarize model
    model.summary()
    plot_model(model, to_file='model_EfficientNetV2M_gru2.png', show_shapes=True)
    return model
