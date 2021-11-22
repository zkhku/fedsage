import os
from stellargraph.core import StellarGraph
from stellargraph.layer import GraphSAGE
from keras.losses import categorical_crossentropy
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from src.utils import config
import dill as pickle
from stellargraph.mapper.sampled_node_generators import GraphSAGENodeGenerator



class Global:
    def __init__(self, globalG: StellarGraph, global_subj, global_targets,test_ids=None):
        self.G = globalG
        self.targets = global_targets
        self.node_subjects = global_subj

        self.info_path = config.server_info_dir
        self.test_acc_path = config.global_test_acc_file
        self.gan_nc_acc_path = config.global_gen_nc_acc_file
        self.globSage=None


        self.org_gen = GraphSAGENodeGenerator(self.G, config.batch_size, config.num_samples)
        self.org_test_gen = self.org_gen.flow(self.node_subjects.index, self.targets)
        if test_ids is not None:
            self.test_ids = test_ids
            self.test_ilocs = self.G.node_ids_to_ilocs(test_ids)
            self.test_only_gen = self.org_gen.flow(self.test_ids, self.targets[self.test_ilocs])
        else:
            self.test_only_gen= None


    def set_test_ids(self,test_ids):
        self.test_ids = test_ids
        self.test_ilocs = self.G.node_ids_to_ilocs(test_ids)
        self.test_only_gen = self.org_gen.flow(self.test_ids, self.targets[self.test_ilocs])

    def set_nc_acc_path(self):
        self.gan_nc_acc_path = config.global_gen_nc_acc_file

    def save_server_info(self):
        if os.path.isfile(self.info_path):
            return
        pickle.dump(self, open(self.info_path, 'wb'))
        return

    def build_glbsage(self):
        graphsage_model = GraphSAGE(layer_sizes=config.classifier_layer_sizes, generator=self.org_gen,
                                    n_samples=config.num_samples)
        x_inp, x_out = graphsage_model.in_out_tensors()
        prediction = Dense(len(self.targets[0]), activation='softmax')(x_out)
        glb_model = Model(inputs=x_inp, outputs=prediction)
        glb_model.compile(
            optimizer=Adam(config.lr),
            loss=categorical_crossentropy,
            metrics=["acc"],
        )
        self.globSage=glb_model

    def train_glbsage(self,train_ids):
        if self.globSage is None:
            self.build_glbsage()
        train_subjects = self.node_subjects[train_ids]
        train_ilocs = self.G.node_ids_to_ilocs(train_ids)
        if os.path.isfile(config.global_classifer_file) is False:
            print("train classifier")
            train_gen = self.org_gen.flow(train_subjects.index, self.targets[train_ilocs], shuffle=True)
            history = self.globSage.fit(
                train_gen, epochs=config.epoch_classifier,
                verbose=2, shuffle=False
            )
            self.globSage.save_weights(config.global_classifer_file)
        else:
            self.globSage.load_weights(config.global_classifer_file)

    def eval_globsage(self):
        glb_test_metrics_glb = self.globSage.evaluate(self.test_only_gen)
        print("\nGlobal model")
        print("\nGlobal Test Set Metrics:")

        with open(config.global_test_acc_file, 'a') as f:
            f.write("\nGlobal model")
            f.write("\nGlobal Test Set Metrics:")
        for name, val in zip(self.globSage.metrics_names, glb_test_metrics_glb):
            print("\t{}: {:0.4f}".format(name, val))
            with open(config.global_test_acc_file, 'a') as f:
                f.write("\t{}: {:0.4f}".format(name, val))

