from stellargraph.core.graph import StellarGraph
from pandas import Series
from stellargraph.mapper import GraphSAGENodeGenerator
from stellargraph.layer import GraphSAGE
import numpy as np
from tensorflow.keras import layers, optimizers, losses, Model
from sklearn import preprocessing, model_selection
import os
import dill as pickle
from src.utils import config,mending_graph


class Classifier:
    def __init__(self,hasG:StellarGraph,
                 all_classes,
                 has_node_subjects:Series,acc_path:str,classifier_path:list,downstream_task_path:str):

        self.downstream_task_path=downstream_task_path
        self.hasG = hasG
        self.has_node_subjects = has_node_subjects


        self.train_subjects, self.test_subjects = model_selection.train_test_split(
            self.has_node_subjects,train_size=0.5,test_size=0.2, stratify=self.has_node_subjects
        )

        self.all_classes=all_classes

        self.train_targets = preprocessing.label_binarize(self.train_subjects,all_classes)
        self.test_targets = preprocessing.label_binarize(self.test_subjects,self.all_classes)
        self.has_node_targets = preprocessing.label_binarize(self.has_node_subjects,self.all_classes)


        self.batch_size = config.batch_size
        self.num_samples = config.num_samples

        self.acc_path=acc_path
        self.classifier_path=classifier_path
        self.feat_shape=len(self.hasG.node_features()[0])

    def set_classifiers(self,classifier_path,dataowner,hasG_hide:StellarGraph):
        self.classifier_path = classifier_path

        self.hasG_hide = hasG_hide
        self.hasG_node_gen = GraphSAGENodeGenerator(self.hasG, self.batch_size, self.num_samples)
        self.all_train_gen = self.hasG_node_gen.flow(self.train_subjects.index, self.train_targets, shuffle=True)
        self.all_test_gen = self.hasG_node_gen.flow(self.test_subjects.index, self.test_targets)

        self.locSage = self.build_classifier(self.hasG_node_gen)
        self.acc_path = dataowner.test_acc_path

        self.fedSage = None
        self.fedSagePC=None

    def save_classifier_instance(self):
        pickle.dump(self, open(self.downstream_task_path, "wb"))
        return



    def build_classifier(self,fillG_node_gen):
        graphsage_model=GraphSAGE(layer_sizes=config.classifier_layer_sizes, generator=fillG_node_gen,
                                  n_samples=config.num_samples)
        x_inp,x_out=graphsage_model.in_out_tensors()
        prediction=layers.Dense(len(self.train_targets[0]),activation='softmax')(x_out)
        model = Model(inputs=x_inp, outputs=prediction)
        model.compile(
            optimizer=optimizers.Adam(config.lr),
            loss=losses.categorical_crossentropy,
            metrics=["acc"],
        )
        return model


    def train_locSage(self):

        history = self.locSage.fit(
            self.all_train_gen, epochs=config.epoch_classifier, verbose=2, shuffle=False
        )
        self.print_test(self.locSage,"LocSage")

    def save_locSage(self):
        print("saving LocSage classifier ...")
        self.locSage.save_weights(self.classifier_path[0])

    def load_locSage(self):
        print("loading LocSage classifier ...")
        self.locSage.load_weights(self.classifier_path[0])

    def print_test(self,classifier,name='LocSage'):
        all_test_metrics_all = classifier.evaluate(self.all_test_gen)
        print("\n"+name)
        print("\nLocal Test Set Metrics:")

        with open(self.acc_path, 'a') as f:
            f.write("\n" + name)
            f.write("\nLocal Test Set Metrics:")
        for name, val in zip(classifier.metrics_names, all_test_metrics_all):
            print("\t{}: {:0.4f}".format(name, val))
            with open(self.acc_path, 'a') as f:
                f.write("\t{}: {:0.4f}".format(name, val))


    def pred_missing_neigh(self, generator_model, all_feat, edges, adj):

        pred_missing, pred_feat, _ = generator_model(all_feat, edges, adj)
        pred_feat.view(-1, config.num_pred, self.feat_shape)
        pred_feat = pred_feat.cpu().detach().numpy()
        pred_missing = np.around(pred_missing.cpu().detach().numpy()).astype(int)

        return pred_missing, pred_feat


    def eval_pred_Gnew(self,generator_model, all_feat, edges,adj, acc_path,
                       impaired_graph:StellarGraph, org_graph:StellarGraph,
                       test_flag=False,save_flag=False,
                       global_task=None):
        pred_missing,pred_feats = \
                self.pred_missing_neigh(generator_model, all_feat, edges,adj)

        if test_flag==True:
            fill_nodes, fill_G = mending_graph.fill_graph(impaired_graph,
                                                          org_graph, pred_missing, pred_feats,
                                                          self.feat_shape)
            fillG_node_gen = GraphSAGENodeGenerator(fill_G, self.batch_size, self.num_samples)
            fill_train_gen = fillG_node_gen.flow(self.train_subjects.index, self.train_targets, shuffle=True)

            self.locSagePC = self.build_classifier(fillG_node_gen)
            if os.path.isfile(self.classifier_path[-1]+"locSagePC.h5") == False:
                history = self.locSagePC.fit(
                    fill_train_gen, epochs=config.epoch_classifier, verbose=2, shuffle=False
                )

                if save_flag:
                    self.locSagePC.save_weights(self.classifier_path[-1]+"locSagePC.h5")
            else:
                self.locSagePC.load_weights(self.classifier_path[-1]+"locSagePC.h5")
            if global_task.test_only_gen is not None:
                self.test_global(global_task, self.locSagePC, acc_path, "LocSagePlusC", "")
                self.test_global(global_task, self.locSage, acc_path, "LocSage", "")

            with open(acc_path, 'a') as f:
                f.write("\nlocal #nodes = " + str(len(self.hasG.node_features())))
                f.write("\nlocal #edges = " + str(len(list(self.hasG.edges()))) + "\n\n\n")

        return pred_missing,pred_feats


    def test_global(self,global_task,classifier,acc_path,name='MD',prefix=''):
        test_metrics_org_fill = classifier.evaluate(global_task.test_only_gen)

        with open(acc_path, 'a') as f:
            f.write("\n"+prefix+" "+name+" Global Org Test Set Metrics:")
        for name, val in zip(classifier.metrics_names, test_metrics_org_fill):
            with open(acc_path, 'a') as f:
                f.write("\t{}: {:0.4f}".format(name, val))
        return test_metrics_org_fill[-1]



    def save_fedSage(self):
        self.fedSage.save_weights(self.classifier_path[-1]+"fedSage.h5")

    def load_fedSage(self,test_gen):
        self.fedSage=self.build_classifier(test_gen)
        self.fedSage.load_weights(self.classifier_path[-1]+"fedSage.h5")


    def save_fedSagePC(self):
        self.fedSagePC.save_weights(self.classifier_path[-1]+"fedSagePlusC.h5")

    def load_fedSagePC(self,test_gen):
        self.fedSagePC=self.build_classifier(test_gen)
        self.fedSagePC.load_weights(self.classifier_path[-1]+"fedSagePlusC.h5")


