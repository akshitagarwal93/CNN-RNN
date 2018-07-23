from typing import Tuple, List, Dict
from keras.models import Sequential
from keras.layers import Conv1D,Conv2D, MaxPooling2D,Embedding, MaxPooling1D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers import Flatten,Dense,SimpleRNN,Dropout
from keras import optimizers
import keras


def create_toy_rnn(input_shape: tuple,
                   n_outputs: int) -> Tuple[keras.Model, Dict]:
    
    toyrnn=Sequential()
    # adding a simple RNN 
    toyrnn.add(SimpleRNN(222,input_shape=input_shape,return_sequences=True))            
    toyrnn.add(Dense(1,activation='linear'))                                             
    toyrnn.compile(loss='mse',optimizer='adam')


    toyrnn_dict={'verbose':1,'epochs':100,'validation_data':None}

    return toyrnn,toyrnn_dict




def create_mnist_cnn(input_shape: tuple,
                     n_outputs: int) -> Tuple[keras.Model, Dict]:
    
    model = Sequential()
    # adding hidden layers with a moderate kernel size and hidden size 
    model.add(Conv2D(18,kernel_size=(2,2),strides=(1,1),activation='tanh',input_shape=input_shape))    
    model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1)))

    model.add(Conv2D(36,kernel_size=(2,2),strides=(1,1),activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1)))
    
    model.add(Conv2D(54,kernel_size=(2,2),strides=(1,1),activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1)))
    
    model.add(Conv2D(108,kernel_size=(2,2),strides=(1,1),activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1)))
    
    model.add(Conv2D(234,kernel_size=(2,2),strides=(1,1),activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=(1,1)))
    
    
    model.add(Flatten())
    
    model.add(Dense(402,activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(702,activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(n_outputs, activation='softmax'))
    sgd=optimizers.SGD(lr=0.01)
    model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])

    model_dict={'verbose':1,'epochs':100,'validation_data':None}



    return model,model_dict


def create_youtube_comment_rnn(vocabulary: List[str],
                               n_outputs: int) -> Tuple[keras.Model, Dict]:
    
    yt_rnn=Sequential()
    
    yt_rnn.add(Embedding(len(vocabulary),100,input_length=200))
    
    yt_rnn.add(SimpleRNN(29,activation='tanh',go_backwards=True))
    
    yt_rnn.add(Dense(36, activation='tanh'))

    yt_rnn.add(Dense(236,activation='tanh'))

    yt_rnn.add(Dense(n_outputs,activation='sigmoid'))
    
    
    adam=optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)                             
    yt_rnn.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])   
    yt_rnn_dict={'verbose':1,'epochs':100,'validation_data':None}
    

    return yt_rnn,yt_rnn_dict

def create_youtube_comment_cnn(vocabulary: List[str],
                               n_outputs: int) -> Tuple[keras.Model, Dict]:
    
    yt_cnn=Sequential()
    yt_cnn.add(Embedding(len(vocabulary),20,input_length=200))
    # adding layers to make the network adjust properly 
    yt_cnn.add(Conv1D(23,kernel_size=3,strides=1,activation='relu'))    
    
    yt_cnn.add(Conv1D(42,kernel_size=3,strides=1,activation='relu'))
    

    yt_cnn.add(Conv1D(62,kernel_size=3,strides=1,activation='relu'))
    
    yt_cnn.add(GlobalMaxPooling1D())
    yt_cnn.add(Dense(12,activation='sigmoid'))
    yt_cnn.add(Dense(n_outputs, activation='sigmoid'))
    adam=optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    yt_cnn.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    yt_cnn_dict={'verbose':1,'epochs':100,'validation_data':None}




    return yt_cnn,yt_cnn_dict


    

