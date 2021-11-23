import torch
root_path= "change_to_your_current_path/"

seed=2021
no_cuda=False
cuda = not no_cuda and torch.cuda.is_available()

dataset = "cora"
num_owners = 3
delta=20

num_samples = [5,5]
batch_size = 32
latent_dim=128
steps=10
epochs_local=1
lr=0.001
weight_decay=1e-4
hidden=32
dropout=0.5

gen_epochs=20
num_pred=5
hidden_portion=0.5

epoch_classifier=20
classifier_layer_sizes=[64,32]

local_test_acc_dir = root_path+'local_result/test_acc/' + dataset+"_"+str(num_owners)
global_test_acc_file = root_path+'global_result/test_acc/' + dataset +"_"+str(num_owners)+".txt"
global_classifer_file=root_path+"global_result/model/"+dataset+"_"+str(num_owners)+"classifier.h5"
local_gen_dir = root_path+'local_result/model/'+dataset+"_"+str(num_owners)
global_gen_nc_acc_file = root_path+'global_result/test_acc/' + dataset +"_nc_"+str(num_owners)+".txt"
local_downstream_task_dir=root_path+'local_result/classifier_info/' + dataset+"_"+str(num_owners)
server_info_dir=root_path +"global_result/server_info/" + dataset+"_"+str(num_owners)+'.h5'
local_dataowner_info_dir=root_path +"dataowner/"+ dataset+"_"+str(num_owners)

a=1
b=1
c=1
