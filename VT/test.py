from vit import Vit
from train import load_data, tf_dataset
import configuration                                                     #import configuration
import numpy as np
import tensorflow as tf
import os
from keras.callbacks import ModelCheckpoint, CSVLogger

print (configuration.cf)
image_size    = configuration.cf["image_size"]
color_channel = configuration.cf["color_channel"]
patch_size    = configuration.cf["patch_size"]
num_patches   = configuration.cf["num_patches"]
flat_patches  = configuration.cf["flat_patches"]
batch_size    = configuration.cf["batch_size"]
learning_rate = configuration.cf["learning_rate"]
epochs        = configuration.cf["epochs"]
num_classes   = configuration.cf["num_classes"]
class_names   = configuration.cf["class_names"]

num_layers = configuration.cf["num_layers"]  
hidden_dim = configuration.cf["hidden_dim"]   
mlp_dim = configuration.cf["mlp_dim"]       
number_heads = configuration.cf["number_heads"]  
dropout_rate = configuration.cf["dropout_rate"]  
 
print(num_patches)


if __name__ == "__main__":

    """Seeding"""
    np.random.seed(42)
    tf.random.set_seed(42)


    """Paths"""
    dataset_path = "/Users/andry/Desktop/VT_data"
    model_path = os.path.join("files", "model.h5")

    """Dataset"""
    train_x, valid_x ,test_x = load_data(dataset_path)
    print(f"Train:{len(train_x)} - Valid:{len(valid_x)} - Test:{len(test_x)}")

    test_ds = tf_dataset(test_x, batch=batch_size)

    """Model"""
    model = Vit(configuration.cf)                                                               #create istance using configuration.cf
    model.load_weights(model_path)                                                              #load weights pre-train

    model.compile(
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False),                      #from_logits=False specifies that the model output has already passed through a softmax activation function, so no further transformations are needed
        optimizer = tf.keras.optimizers.Adam(learning_rate),
        metrics=["acc"]
    )

    csv_folder = "/Users/andry/Desktop/VT/files"                                                #Specify the path to the VT folder
    csv_filename = "log_test.csv"
    csv_path = os.path.join(csv_folder, csv_filename)

    callbacks = [
        CSVLogger(csv_path),
    ]

    loss, accuracy = model.evaluate(test_ds)                                                    #The evaluate() function is used to evaluate the performance of the model on a set of test data.

    #Manually save test results to log file
    with open(csv_path, "a") as file:
        file.write("{:<10}{:<10}\n".format("Accuracy_test", "Loss_test"))
        file.write("{:<10.4f}{:<10.4f}\n".format( accuracy, loss))

    

