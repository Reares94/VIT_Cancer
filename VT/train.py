
from sklearn.utils import shuffle                                                                   # shuffling the elements of an array in the training case before passing them to the algorithm
from glob import glob                                                                               # glob is for getting all files with a specific extension in a given directory
import numpy as np                                                                                  # for multidimensional arrays and scientific calculations
import cv2                                                                                          # It provides functionality to upload, edit and save images

import os                                                                                           #browsing directories, running system commands, and more
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"                                                              #reduce the amount of output in the terminal


import configuration                                                                                #import configuration.py


from sklearn.model_selection import train_test_split                                                #Split data
from patchify import patchify                                                                       #It is used to convert the image to patch and then flatten and then input to the network
import tensorflow as tf                                                                             #import tensorflow
import pathlib                                                                                      #manipulation of file and directory paths
#from skimage.util import patchify
#from vit import Vit
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping            #callback
import pandas as pd                                                                                 #data analysis
import matplotlib.pyplot as plt                                                                     #data visualization
import shutil                                                                                       #file and directory operations, such as copying, moving, and deleting.


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


#Global var

IMAGE_PATH = '/Users/andry/Desktop/Histological_Cancer/train/'
SAMPLE_SIZE =80000
base_dir= "/Users/andry/Desktop/VT_data"
path = "/Users/andry/Desktop/VT/"

#Functions 

#1) Create Directory                                                                                             #creates a new directory in the path of the current user's desktop folder
def create_folder(folder_name):
    desktop_path = os.path.expanduser(path)                                                                      #initialize the path to the current user's desktop folder using the os.path.expanduser(path)                                                                        
    folder_path = os.path.join(desktop_path, folder_name)
    pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)                                                 #parents=True and exist_ok=True parameters indicate that parent paths should also be created if they don't exist and that no exception should be raised if the directory already exists.

#2) Draw category images
def draw_category_images(col_name, figure_cols, df, IMAGE_PATH):

    categories = (df.groupby([col_name]).nunique()).index                                                        #Gets the list of categories in the df dataframe by grouping the unique values of the col_name column and assigning them to the categories variable.

    f, ax = plt.subplots(nrows=len(categories),ncols=figure_cols,                                                #Create a figure and axes for the graph
                         figsize=(4*figure_cols,4*len(categories))) 

    # draw a number of images for each location
    for i, cat in enumerate(categories):
        sample = df[df[col_name]==cat].sample(figure_cols)                                                       #randomly extract figure cols froma dataframe of that match that category and assigns them to variable sample            
        for j in range(0,figure_cols):
            
            file= IMAGE_PATH + sample.iloc[j]['id'] + '.tif'                                                     #load the image using Image path with image id taken from the dataframe
            im=cv2.imread(file)
            ax[i, j].imshow(im, resample=True)
            ax[i, j].set_title('Class ->'+ str(cat), fontsize=8)  
    plt.tight_layout()
    plt.show()

#3) Prepare data
def prepare_data(data):
    df_train, df_val = train_test_split(data, test_size=0.10, random_state=101)
    print('df_train : '+ str(df_train.shape))
    print('df_val : ' + str(df_val.shape))
    
    return df_train, df_val

#4) Create dictionary
def create_dictionary(base_dir):                                                                                    #The create_dictionary function takes a base directory base_dir as input and creates a set of subdirectories within it.
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)

        train_dir_NO_tumor = os.path.join(base_dir, 'train_dir_NO_tumor')
        os.mkdir(train_dir_NO_tumor)
            
        train_dir_YES_tumor = os.path.join(base_dir, 'train_dir_YES_tumor')
        os.mkdir(train_dir_YES_tumor)

        val_dir_NO_tumor = os.path.join(base_dir, 'val_dir_NO_tumor')
        os.mkdir(val_dir_NO_tumor)

        val_dir_YES_tumor = os.path.join(base_dir, 'val_dir_YES_tumor')
        os.mkdir(val_dir_YES_tumor)

#5) Add images in dictionary
def  add_image_folder_dictionary(train_list, val_list):
        if not os.path.exists(base_dir):                                                                            #If it doesn't exist, the code block inside the conditional block is executed.
            for image in train_list:
                fname = image + '.tif'
                target = df_data.loc[image, 'label']                                                                #assigns the target variable the value corresponding to the colum label in dataframe 
                
                if target == 0:
                    label = 'train_dir_NO_tumor'
                if target == 1:
                    label = 'train_dir_YES_tumor'

                src = os.path.join('/Users/andry/Desktop/Histological_Cancer/train', fname)
                dst = os.path.join(base_dir, label, fname)

                shutil.copyfile(src, dst)                                                                         

            for image in val_list:
                fname = image + '.tif'
                target = df_data.loc[image, 'label']

                if target == 0:
                    label = 'val_dir_NO_tumor'
                if target == 1:
                    label = 'val_dir_YES_tumor'

                src = os.path.join('/Users/andry/Desktop/Histological_Cancer/train', fname)
                dst = os.path.join(base_dir, label, fname)

                shutil.copyfile(src,dst)

#6) Check train and val
def check_images():
    print("Train_NO_tumor  ---  "+str(len(os.listdir('/Users/andry/Desktop/VT_data/train_dir_NO_tumor'))))
    print("Train_YES_tumor ---  "+str(len(os.listdir('/Users/andry/Desktop/VT_data/train_dir_YES_tumor'))))
    print("Val_NO_tumor    ---  "+str(len(os.listdir('/Users/andry/Desktop/VT_data/val_dir_NO_tumor'))))
    print("Val_Yes_tumor   ---  "+str(len(os.listdir('/Users/andry/Desktop/VT_data/val_dir_YES_tumor'))))

#7) Iterable and Data split
def load_data(path, split=0.1):                                                                                     #10% for testing and another for validation 80 training, 10 tests, 10 validations
    images = shuffle(glob(os.path.join(path, "*", "*.tif")))                                                        #first give the path, then you say from all the folders, then extract the images. Shuffle to make the model more performing but maybe slower
    #print(images)
    split_size =  int(len(images) * split)                                                                          #calculates the size of the data split by a specified proportion.
    print("Lenght images:",len(images))       
    train_x, valid_x = train_test_split(images, test_size = split_size, random_state =42 )    
    train_x, test_x = train_test_split(train_x, test_size = split_size, random_state =42 ) 
    return train_x, test_x, valid_x

#8) Process Image
def process_image_label(path):
    """Reading images"""
    path = path.decode()
    image = cv2.imread(path, cv2.IMREAD_COLOR)                                                                      #Read the image specified by the path and load it as a color image
    image = cv2.resize(image, (image_size, image_size)) 
    #print("Image shape: ", image.shape)

    """ Preprocessing to patches """
    patch_shape = (patch_size, patch_size, color_channel)
    patches = patchify(image, patch_shape, patch_size)
    #print("Dimension patch: ", patches.shape)
    patches = np.reshape(patches,(16, 24, 24, 3))
    #print("Reshape dimension patch: ", patches.shape)
    #---------------------------------------------------------------------
    #files_path = '/Users/andry/Desktop/VT/files'
    #for i in range(16):
    #    cv2.imwrite(os.path.join(files_path,f"files{i}.jpg"), patches[i])
    #---------------------------------------------------------------------
    """Flatten Patch(16,1728)"""
    patches = np.reshape(patches, flat_patches)
    patches = patches.astype(np.float32)
    #print("Flatten dimension patch: ",patches.shape)

    """Label class""" 
    #print(path)
    class_name = path.split("/")
    #print(class_name)
    class_name = path.split("/")[-2]                                                                                #it is updated to contain only the last item in the list, i.e. the file name.
    #print(class_name)
    class_idx = class_names.index(class_name)                                                                       #The class index corresponding to the file name is obtained using class_names.index(class_name)
    #print(class_idx)
    class_idx = np.array(class_idx, dtype= np.int32)                                                                #Finally, class_idx is converted into an array of type np.int32 and returned together with patches.
    return patches, class_idx
    
   
#9)Parse function
def parse(path):
    patches, labels = tf.numpy_function(process_image_label, [path], [tf.float32, tf.int32])                        #patches and class_idx are assigned to the patches and labels variables  
    labels = tf.one_hot(labels, num_classes)                                                                        #labels are converted to one-hot encoding

    patches.set_shape(flat_patches)                                                                                 #patches is set to flat_patches, which represents the desired shape of the flat patches                                                            
    labels.set_shape(num_classes)                                                                                   #labels is set to num_classes, which represents the number of classes in the problem.

    return patches, labels

#10)Dataset 
def tf_dataset(images, batch = batch_size):                                                                         #The tf_dataset function creates a TensorFlow dataset object from an array of images
    ds = tf.data.Dataset.from_tensor_slices(images)                                                                 #The from_tensor_slices method will create a dataset where each element of images is treated as a separate element of the dataset
    ds = ds.map(parse).batch(batch).prefetch(8)                                                                     #The map method is used to apply the parse function.The dataset is batched using batch.Prefetch the data
    return ds

    

if __name__ == "__main__":

    """ Seeding """
    np.random.seed(42)                                              
    tf.random.set_seed(42)                                                                                          #import seed for Numpy's random number generator

    """Pre_elaboration_data"""
    create_folder("files")                                                                                          #create folder files 

    """Path"""
    model_path =  os.path.join("/Users/andry/Desktop/VT/files", "model.h5")                                         #the paths for the model.h5 
    csv_path = os.path.join("/Users/andry/Desktop/VT/files", "log.csv")                                             #and log files are created
   
    """Draw_category_images"""
    df_data = pd.read_csv("/Users/andry/Desktop/Histological_Cancer/train_labels.csv")                              #Read Data
    print("La forma del dataframe = "+str(df_data.shape))                                                           
    draw_category_images('label', 6, df_data, IMAGE_PATH)                                                           #Show image dataframe 

    """DataFrame"""
    df_0 = df_data[df_data['label'] == 0].sample(SAMPLE_SIZE, random_state = 101)                   
    df_1 = df_data[df_data['label'] == 1].sample(SAMPLE_SIZE, random_state = 101)                                   #select the rows of the data where the column is equal to label
    df_data = pd.concat([df_0, df_1], axis=0).reset_index(drop=True)                                                #concatenate dataframes and reset indexes continuously discarding previous ones
    df_data = shuffle(df_data)                                                                                      #shuffle data
    print(df_data['label'].value_counts())

    """Prepare data """
    df_train, df_val = prepare_data(df_data)                                                          
    df_data = shuffle(df_data)                                                                          
    print(df_data['label'].value_counts())              

    """Create dictionary structure"""
    create_dictionary(base_dir)                                                                                     #create dictionary

    """Set the index"""
    df_data.set_index('id', inplace=True)                                                                           #Set the id column as the index of the dataframe
    train_list = list(df_train['id'])                                                                               #Create a list called train_list containing the values of the 'id' column of the DataFrame.
    val_list = list(df_val['id'])                                                                                   #The list() method is used to convert the column values into a list.

    """Add images to the dictionary"""
    add_image_folder_dictionary(train_list, val_list)                                                               #add image in folder                         

    """Check data images"""
    check_images()                                                                                  


    dataset_path = "/Users/andry/Desktop/VT_data"

    train_x, test_x, valid_x = load_data(dataset_path) 
    print("Train_X------" +str(len(train_x)))
    print("Test_X-------" +str(len(test_x)))
    print("Valid_X------" +str(len(valid_x)))

 
    train_ds = tf_dataset(train_x, batch_size)
    valid_ds = tf_dataset(valid_x, batch_size)

    from vit import Vit
    model = Vit(configuration.cf)


    model.compile(
        loss = "categorical_crossentropy",                                                                          #loss function
        optimizer = tf.keras.optimizers.Adam(learning_rate, clipvalue=1.0),                                         #specifies the optimizer
        metrics = ["acc"]
    )

    callbacks = [
        ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True),                            #save your weights when the loss validation decreases
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-10, verbose=1),                     #a plateau alarm that reduces the learning rate when the loss validation stops falling
        CSVLogger(csv_path),                                                                                        #create a log file that keeps us track of accuracy and everything
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False),                                 #manages a stop mechanism if after 50 epochs the loss does not go down 
    ]
    
    model.fit(
        train_ds,
        epochs=epochs,
        validation_data=valid_ds,
        callbacks=callbacks
    )

    


