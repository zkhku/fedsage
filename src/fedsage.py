from src.utils import config
from src.global_task import Global
from src.classifiers import Classifier
from stellargraph.mapper import GraphSAGENodeGenerator

def train_fedSage(classifier_list: list, global_task: Global, acc_path):
    assert len(classifier_list) == config.num_owners
    for classifier in classifier_list:
        assert classifier.__class__.__name__ == Classifier.__name__
    all_list = []
    for classifier in classifier_list:
        classifier.fedSage = classifier.build_classifier(classifier.hasG_node_gen)
        all_list.append(classifier.fedSage)
    weights = all_list[0].get_weights()
    weights_len = len(weights)
    for all_classifier in all_list[1:]:
        weights_cur = all_classifier.get_weights()
        for i in range(weights_len):
            weights[i] += weights_cur[i]
    for i in range(weights_len):
        weights[i] = 1.0 / config.num_owners * weights[i]
    train_node_gen_list = []
    for classifier in classifier_list:
        classifier.fedSage.set_weights(weights)
        hasG_node_gen = GraphSAGENodeGenerator(classifier.hasG, config.batch_size, classifier.num_samples)
        all_train_gen = hasG_node_gen.flow(classifier.train_subjects.index, classifier.train_targets, shuffle=True)
        train_node_gen_list.append(all_train_gen)
    grad_list = []
    classifier = classifier_list[0]
    for epoch in range(config.epoch_classifier):
        weight_cur = classifier.fedSage.get_weights()
        for classifier_i in range(len(classifier_list)):
            history = classifier.fedSage.fit(train_node_gen_list[classifier_i],
                                                         epochs=config.epochs_local,
                                                         verbose=2, shuffle=False)
            weight_send = classifier.fedSage.get_weights()
            grad_list.append([weight_send])
            classifier.fedSage.set_weights(weight_cur)
            print("local do = " + str(classifier_i) + " communication round = " + str(epoch))

        for grad in grad_list[1:]:
            for i in range(len(grad[0])):
                grad_list[0][0][i] += grad[0][i]
        for i in range(len(grad_list[0][0])):
            grad_list[0][0][i] *= 1.0 / config.num_owners
        classifier.fedSage.set_weights(grad_list[0][0])
        print("epoch " + str(epoch))
        grad_list = []

    print("FedSage end!")

    with open(acc_path, 'a') as f:
        f.write("\nFedSage")
    classifier.save_fedSage()
    classifier.load_fedSage(global_task.org_gen)
    classifier.test_global(global_task,
                           classifier.fedSage, acc_path,
                           name='FedSage', prefix='')

    return
