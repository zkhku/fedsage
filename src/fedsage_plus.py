from src.train_locSagePlus import LocalOwner
from src.utils import config
import torch
import torch.nn.functional as F
from torch import optim
from src.global_task import Global
from src.models import feat_loss
from src.utils import mending_graph
from stellargraph.mapper import GraphSAGENodeGenerator
import numpy as np
import time

def train_fedgen(local_owners:list,feat_shape:int):
    assert len(local_owners) == config.num_owners
    for owner in local_owners:
        assert owner.__class__.__name__ == LocalOwner.__name__
    local_gen_list=[]
    optim_list=[]
    t=time.time()
    for local_i in local_owners:
        local_i.set_fed_model()
        local_gen_list.append(local_i.fed_model.gen)
        optim_list.append(optim.Adam(local_gen_list[-1].parameters(),
                                  lr=config.lr, weight_decay=config.weight_decay))
    for epoch in range(config.gen_epochs):
        for i in range(config.num_owners):
            local_gen_list[i].train()
            optim_list[i].zero_grad()
            local_model=local_owners[i].fed_model
            input_feat = local_owners[i].all_feat
            input_edge = local_owners[i].edges
            input_adj = local_owners[i].adj
            output_missing, output_feat, output_nc = local_model(input_feat, input_edge, input_adj)
            output_missing = torch.flatten(output_missing)
            output_feat = output_feat.view(len(local_owners[i].all_ids), local_owners[i].num_pred, local_owners[i].feat_shape)
            output_nc = output_nc.view(len(local_owners[i].all_ids) + len(local_owners[i].all_ids) * local_owners[i].num_pred, local_owners[i].num_classes)
            loss_train_missing = F.smooth_l1_loss(output_missing[local_owners[i].train_ilocs].float(),
                                                  local_owners[i].all_targets_missing[local_owners[i].
                                                  train_ilocs].reshape(-1).float())

            loss_train_feat = feat_loss.greedy_loss(output_feat[local_owners[i].train_ilocs],
                                               local_owners[i].all_targets_feat[local_owners[i].train_ilocs],
                                               output_missing[local_owners[i].train_ilocs],
                                               local_owners[i].all_targets_missing[
                                                    local_owners[i].train_ilocs
                                                ]).unsqueeze(0).mean().float()
            true_nc_label = torch.argmax(local_owners[i].all_targets_subj[local_owners[i].train_ilocs], dim=1).view(-1)
            if config.cuda:
                true_nc_label = true_nc_label.cuda()
            loss_train_label = F.cross_entropy(output_nc[local_owners[i].train_ilocs], true_nc_label)

            acc_train_missing = local_owners[i].accuracy_missing(output_missing[local_owners[i].train_ilocs],
                                                      local_owners[i].all_targets_missing[local_owners[i].train_ilocs])
            acc_train_nc = local_owners[i].accuracy(output_nc[local_owners[i].train_ilocs],
                                         local_owners[i].all_targets_subj[local_owners[i].train_ilocs])


            loss = (config.a * loss_train_missing + config.b * loss_train_feat + config.c * loss_train_label).float()
            print('Data owner ' + str(i),
                  ' Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss.item()),
                  'missing_train: {:.4f}'.format(acc_train_missing),
                  'nc_train: {:.4f}'.format(acc_train_nc),
                  'loss_miss: {:.4f}'.format(loss_train_missing.item()),
                  'loss_nc: {:.4f}'.format(loss_train_label.item()),
                  'loss_feat: {:.4f}'.format(loss_train_feat.item()),
                  'time: {:.4f}s'.format(time.time() - t))

            for j in range(config.num_owners):
                if j != i:
                    choice = np.random.choice(len(list(local_owners[j].subG.nodes())),
                                              len(local_owners[i].train_ilocs))
                    others_ids=local_owners[j].subG.nodes()[choice]
                    global_target_feat = []
                    for c_i in others_ids:
                        neighbors_ids=local_owners[j].subG.neighbors(c_i)
                        while len(neighbors_ids)==0:
                            c_i=np.random.choice(len(list(local_owners[j].subG.nodes())),1)[0]
                            id_i = local_owners[j].subG.nodes()[c_i]
                            neighbors_ids = local_owners[j].subG.neighbors(id_i)
                        choice_i = np.random.choice(neighbors_ids,config.num_pred)
                        for ch_i in choice_i:
                            global_target_feat.append(local_owners[j].subG.node_features([ch_i])[0])
                    global_target_feat = np.asarray(global_target_feat).reshape(
                        (len(local_owners[i].train_ilocs), config.num_pred, feat_shape))
                    loss_train_feat_other = feat_loss.greedy_loss(output_feat[local_owners[i].train_ilocs],
                                                             global_target_feat,
                                                             output_missing[local_owners[i].train_ilocs],
                                                             local_owners[i].all_targets_missing[
                                                                  local_owners[i].train_ilocs]
                                                             ).unsqueeze(0).mean().float()
                    loss += config.b * loss_train_feat_other
            loss = 1.0 / config.num_owners * loss
            loss.backward()
            optim_list[i].step()

    for i in range(config.num_owners):
        local_owners[i].save_fed_model()

    return



def train_fedSagePC(classifier_list:list,local_owner_list:list,global_task:Global,acc_path):
    assert len(classifier_list) == config.num_owners
    fed_gen_classifier_list=[]
    fill_train_gen_list=[]
    for classifier in classifier_list:
        classifier.fedSagePC=classifier.build_classifier(classifier.hasG_node_gen)
        fed_gen_classifier_list.append(classifier.fedSagePC)
    weights=fed_gen_classifier_list[0].get_weights()
    weights_len=len(weights)
    for fed_gen_classifier in fed_gen_classifier_list[1:]:
        weights_cur=fed_gen_classifier.get_weights()
        for i in range(weights_len):
            weights[i]+=weights_cur[i]
    for i in range(weights_len):
        weights[i]=1.0/config.num_owners*weights[i]
    for owner_i in range(config.num_owners):
        local_owner = local_owner_list[owner_i]
        classifier = classifier_list[owner_i]
        input_feat = local_owner.all_feat
        input_edge = local_owner.edges
        input_adj = local_owner.adj
        pred_missing, pred_feats, _ = local_owner.fed_model(input_feat, input_edge, input_adj)

        fill_nodes, fill_G = mending_graph.fill_graph(local_owner.hasG_hide,
                                                      local_owner.subG,
                                                      pred_missing, pred_feats, local_owner.feat_shape)

        fillG_node_gen = GraphSAGENodeGenerator(fill_G, config.batch_size, config.num_samples)
        fill_train_gen = fillG_node_gen.flow(classifier.train_subjects.index, classifier.train_targets, shuffle=True)
        fill_train_gen_list.append(fill_train_gen)
        classifier.fedSagePC = classifier.build_classifier(fillG_node_gen)
        classifier.fedSagePC.set_weights(weights)
    grad_list=[]
    classifier=classifier_list[0]

    for epoch in range(config.epoch_classifier):
        weight_cur = classifier.fedSagePC.get_weights()
        for owner_i in range(config.num_owners):
            history = classifier.fedSagePC.fit(fill_train_gen_list[owner_i],
                                                         epochs=config.epochs_local,
                                                         verbose=2, shuffle=False)
            weight_send = classifier.fedSagePC.get_weights()
            grad_list.append([weight_send])
            classifier.fedSagePC.set_weights(weight_cur)
            print("local do = " + str(owner_i) + " communication round = " + str(epoch))

        for grad in grad_list[1:]:
            for i in range(len(grad[0])):
                grad_list[0][0][i] += grad[0][i]
        for i in range(len(grad_list[0][0])):
            grad_list[0][0][i] *= 1.0 / config.num_owners
        classifier.fedSagePC.set_weights(grad_list[0][0])
        print("epoch " + str(epoch))
        grad_list = []

    print("FedSage+ end!")
    classifier.save_fedSagePC()
    classifier.load_fedSagePC(global_task.org_gen)
    classifier.test_global(global_task,classifier.fedSagePC,acc_path,
                               name='FedSage+',prefix='')

    return
