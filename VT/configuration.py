
cf = {}                                                                                                     #create an empty dictionary
cf["image_size"] = 96                                                                                       #Image size 96x96  
cf["color_channel"] = 3                                                                                     #RGB
cf["patch_size"] = 24                                                                                       #must be a multiple of the image size
cf["num_patches"] = (cf["image_size"]**2) // (cf["patch_size"]**2)                                          # N = HW / P^{2} = 16
cf["flat_patches"] = (cf["num_patches"], (cf["patch_size"]**2)*cf["color_channel"])                         # N x (P^{2}*C)  = (16, 1875)

#----------Setting these values------------
cf["epochs"] = 10                                                                                           #number of eras
cf["batch_size"] = 128                                                                                      #you manage the batch value based on the power of the gpu (16GB of Ram)
cf["learning_rate"] = 1e-3                                                                                  #smaller and more accurate
#------------------------------------------


cf["num_classes"] = 4
cf["class_names"] = ["train_dir_NO_tumor", "train_dir_YES_tumor", "val_dir_NO_tumor", "val_dir_YES_tumor"]

#-----------Choose the model--------------
modello_scelto = "vit_base"                                                                   

if modello_scelto == "vit_base":
    cf["num_layers"] = 12
    cf["hidden_dim"] = 768                                                                                  #D - paper
    cf["mlp_dim"] = 3072
    cf["number_heads"] = 12
    cf["dropout_rate"] = 0.1
elif modello_scelto == "vit_mezzo_base":
    cf["num_layers"] = 6
    cf["hidden_dim"] = 384
    cf["mlp_dim"] = 1536
    cf["number_heads"] = 6
    cf["dropout_rate"] = 0.1
elif modello_scelto == "vit_unquarto_base":
    cf["num_layers"] = 3
    cf["hidden_dim"] = 192
    cf["mlp_dim"] = 768
    cf["number_heads"] = 3
    cf["dropout_rate"] = 0.1
else:
    print("Modello non valido")

#------------------------------------------
"""""
#Series of experiments I want to do:
##########  case1  case2   case3        case1   case2   case3               case1    case2    case3
_________|      #mod base        |           #mod half               |           #mod quarter          |      
#epoche      5     10       20   |         5     10      20          |           5     10      20      |
#batch       64    128     256   |        64    128     256          |          64    128     256      |
#lr          1e-2  1e-3   1e-4   |      1e-2   1e-3    1e-4          |        1e-2   1e-3    1e-4      |

"""""
