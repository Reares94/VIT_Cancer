import os 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"                                                                            #reduce message output

import tensorflow as tf                                                                                             #import TensorFlow
from keras.layers import *                                                                                          #import all layers available in the keras.layers module.
from keras.models import Model
import configuration


num_layers     = configuration.cf["num_layers"]
hidden_dim     = configuration.cf["hidden_dim"]
mlp_dim        = configuration.cf["mlp_dim"]
number_heads   = configuration.cf["number_heads"]
dropout_rate   = configuration.cf["dropout_rate"]
num_patches    = configuration.cf["num_patches"]
patch_size     = configuration.cf["patch_size"]
color_channels = configuration.cf["color_channel"]
num_classes    = configuration.cf["num_classes"]

print(configuration.cf)

#Funzione Input Layer 
def input_layer(num_patches,patch_size,color_channels):
    input_shape = (num_patches, (patch_size**2)*color_channels)                 
    inputs = Input(input_shape)                                                                                     #The function creates an Input object using the shape of the input
    print("Input Dimension:", inputs.shape)                                                                  
    return inputs

#Patch + Position Embedding 
def patch_position_embedding(hidden_dim, inputs):
    """Patch + Position Embedding"""                                                                                
    patch_embedding = Dense(hidden_dim)(inputs)                                                                     #I apply Dense layer fully connected to input inputs,  
    print("Patch Embedding:", patch_embedding.shape)                                                                #(None, 16,768) 
    positions = tf.range(start = 0, limit = num_patches,delta =1)                                                   #positional embedding
    print("Position:", positions)
    position_embedding = Embedding(input_dim=num_patches, output_dim= hidden_dim )(positions)                       #The Embedding layer assigns a unique embedding to each input location.
    print("Position Embedding:", position_embedding.shape)
    p_embed = patch_embedding + position_embedding
    print("Patch + Position", p_embed.shape)
    return p_embed

class AddToken(Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        weights_initialization = tf.random_normal_initializer()                                                     
        self.w = tf.Variable(
            initial_value = weights_initialization(shape=(1,1,input_shape[-1]), dtype=tf.float32),                  #initialized a variable self.w as a tensor
            trainable = True                                                                                        #dynamically adapt the embedding
        )
        print("Input_shape dimension", input_shape)
  
    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]                                                                            #obtain batch dimension
        hidden_dim =self.w.shape[-1]                                                                                #obtain hidden dim from self.w
        cls = tf.broadcast_to(self.w, [batch_size,1,hidden_dim])                                                    #tf.broadcast_to to expand self.w to a size of shape [batch_size, 1, hidden_dim]
        cls = tf.cast(cls, dtype=inputs.dtype)                                                                      #tf.cast to ensure that the data type of cls matches that of inputs.
        return cls
    
def mlp(input):                                                                                                     #applies a series of transformations using densely connected (fully connected) and dropout layers.
    input = Dense(mlp_dim, activation="gelu")(input)
    input = Dropout(dropout_rate)(input)                  
    input = Dense(hidden_dim)(input)                                                                                
    input = Dropout(dropout_rate)(input)  
    return input

    
def transformer_encoder(input_transformer):                                                                         #The transformer_encoder function implements a single block of an encoder of the Transformer model.
    #First Jump
    first_jump = input_transformer
    #Layer Normalization
    input_transformer = LayerNormalization()(input_transformer)                                                     #LN
    #MultiHeadAttention
    input_transformer = MultiHeadAttention(                                                                         #Multi Head
        num_heads= number_heads , key_dim= hidden_dim
    )(input_transformer,input_transformer)
    #Adding jump
    input_transformer = Add()([input_transformer, first_jump])                                                      #Residual connection
    #Second Jump
    second_jump = input_transformer
    #Second Layer Normalization
    input_transformer = LayerNormalization()(input_transformer)                                                     #LN
    #Mlp    
    input_transformer = mlp(input_transformer)                                                                      #MLP
    #Adding second Jump
    input_transformer = Add()([input_transformer, second_jump])                                                     #Second residual connection
    #La dimensione in ingresso deve essere come quella in uscita
    return input_transformer

def Vit(cf):
    """Input Layer"""
    inputs = input_layer(num_patches,patch_size,color_channels)                                                     #Create the input layer of the model
    print("Input dimension:", inputs.shape)   

    """Patch + Position Embedding """
    embed = patch_position_embedding(hidden_dim,inputs)                                                             #Apply patch and position embedding
    print(" Patch + position dimension:", embed.shape)   

    """Add Token"""
    token = AddToken()(embed)                                                                                       #Adds the special token using the AddToken class
    x = Concatenate(axis=1)([token, embed])                                                                            
    print("Token dimension: ", x.shape)

    """Repeat N-times block Encoder"""
    for _ in range(num_layers):                 
        x = transformer_encoder(x)                                                                                  #Repeat the encoder block

    """Classification"""
    print("Output shape:", x.shape)
    x = LayerNormalization()(x)
    x = x[:,0,:]                                                                                                    #Extracts the first position of the final output as a representation
    print("x_shape:", x.shape)
    x = Dense(num_classes,activation="softmax")(x)                                                                  #Applies a densely connected layer with softmax activation for classification.

    model = Model(inputs,x)
    return model

if __name__ == "__main__":

    model = Vit(configuration.cf)
    model.summary()
    