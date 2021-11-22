from src.utils import config
from src.utils.louvain_networkx import louvain_graph_cut
from src.utils.load_ms import load_from_npz
from stellargraph import datasets
from src.dataowner import DataOwner
from src.classifiers import Classifier
from src.train_locSagePlus import LocalOwner
from src.fedsage import train_fedSage
from src.global_task import Global
from sklearn import preprocessing
from src.fedsage_plus import train_fedgen,train_fedSagePC


def set_up_system():
    if config.dataset == 'cora':
        dataset = datasets.Cora()
        G, node_subjects = dataset.load()
    elif config.dataset == 'citeseer':
        dataset = datasets.CiteSeer()
        G, node_subjects = dataset.load()
    elif config.dataset == 'pubmeddiabetes':
        dataset = datasets.PubMedDiabetes()
        G, node_subjects = dataset.load()
    elif config.dataset == 'msacademic':
        G, node_subjects = load_from_npz(config.root_path + 'other_datasets/ms_academic.npz')
    else:
        print("dataset name does not exist!")
        return

    target_encoding = preprocessing.LabelBinarizer()

    global_targets = target_encoding.fit_transform(node_subjects)

    all_classes=target_encoding.classes_


    global_task=Global(G,node_subjects,global_targets)


    dataowner_list = []
    local_G, local_subj, local_target, local_nodes_ids = louvain_graph_cut(G, node_subjects)

    for owner_i in range(config.num_owners):
        do_i = DataOwner(do_id=owner_i, subG=local_G[owner_i], sub_ids=local_nodes_ids[owner_i],
                         node_subj=local_subj[owner_i],
                         node_target=local_target[owner_i])

        do_i.get_edge_nodes()
        do_i.set_classifier_path()
        do_i.set_gan_path()
        do_i.save_do_info()
        dataowner_list.append(do_i)

    # begin train local pre-train
    local_owners=[]
    local_classifiers=[]
    train_ids = []
    test_ids = []
    for owner_i in range(config.num_owners):
        do_i=dataowner_list[owner_i]
        local_classifier = Classifier(hasG=do_i.hasG,
                                          all_classes=all_classes,
                                          has_node_subjects=do_i.has_subj,
                                          acc_path=do_i.test_acc_path, classifier_path=do_i.classifier_path,
                                          downstream_task_path=do_i.downstream_task_path)


        local_gen = LocalOwner(do_id=owner_i, subG=do_i.hasG, node_subjects=do_i.has_subj,
                               all_classes=all_classes,
                               num_samples=config.num_samples,
                               model_path=[do_i.fedgen_model_path, do_i.gen_model_path])
        local_classifier.set_classifiers(classifier_path=do_i.classifier_path, dataowner=do_i,
                                         hasG_hide=local_gen.hasG_hide)
        print("train classifier for do_i " + str(owner_i))
        local_classifier.train_locSage()


        print("train GAN for do_i " + str(owner_i))
        local_gen.train()
        local_owners.append(local_gen)

        local_classifiers.append(local_classifier)
        for train_id in local_classifier.train_subjects.index:
            train_ids.append(train_id)
        for test_id in local_classifier.test_subjects.index:
            test_ids.append(test_id)


    for do_i in range(config.num_owners):
        test_ids_i=local_classifiers[do_i].test_subjects.index
        for id_i in test_ids_i:
            test_ids.append(id_i)
    global_task.set_test_ids(test_ids)

    for do_i in range(config.num_owners):
        local_owners[do_i].evaluate_downstream(local_classifiers[do_i], test_flag=True, save_flag=True,
                                  global_task=global_task)


    feat_shape=local_owners[0].feat_shape

    train_fedSage(local_classifiers, global_task, config.global_gen_nc_acc_file)


    for owner in local_owners:
        owner.set_fed_model()
    train_fedgen(local_owners,feat_shape)

    train_fedSagePC(local_classifiers, local_owners, global_task, config.global_gen_nc_acc_file)


    global_task.train_glbsage(train_ids)
    global_task.eval_globsage()





set_up_system()
