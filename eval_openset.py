"""
Stand alone evaluation script for open set recognition and plotting of different datasets

Uses the same command line parser as main.py

The attributes that need to be specified are the number of variational samples (should be greater than one if prediction
uncertainties are supposed to be calculated and compared), the architecture type and the resume flag pointing to a model
checkpoint file.
Other parameters like open set distance function etc. are optional.

example usage:
--resume /path/checkpoint.pth.tar --var-samples 100 -a MLP
"""
import collections

from lib.cmdparser import parser
import lib.Datasets.datasets as datasets
import lib.Models.architectures as architectures
from lib.Utility.visualization import *
from lib.OpenSet.meta_recognition import *


def main():
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Command line options
    args = parser.parse_args()
    print("Command line options:")
    for arg in vars(args):
        print(arg, getattr(args, arg))

    # choose dataset evaluation function to import (e.g. variational will operate on z values)
    if args.train_var:
        from lib.Training.evaluate import eval_var_dataset as eval_dataset
        from lib.Training.evaluate import eval_var_openset_dataset as eval_openset_dataset
    else:
        from lib.Training.evaluate import eval_dataset as eval_dataset
        from lib.Training.evaluate import eval_openset_dataset as eval_openset_dataset

    # Get the dataset which has been trained and the corresponding number of classes
    data_init_method = getattr(datasets, args.dataset)
    dataset = data_init_method(torch.cuda.is_available(), args)
    num_classes = dataset.num_classes
    net_input, _ = next(iter(dataset.train_loader))
    num_colors = net_input.size(1)

    # Split a part of the non-used dataset to use as validation set for determining open set (e.g entropy)
    # rejection thresholds
    split_perc = 0.5
    split_sets = torch.utils.data.random_split(dataset.valset,
                                               [int((1 - split_perc) * len(dataset.valset)),
                                                int(split_perc * len(dataset.valset))])

    # overwrite old set and create new split set to determine thresholds/priors
    dataset.valset = split_sets[0]
    dataset.threshset = split_sets[1]

    # overwrite old data loader and create new loader for thresh set
    is_gpu = torch.cuda.is_available()
    dataset.val_loader = torch.utils.data.DataLoader(dataset.valset, batch_size=args.batch_size, shuffle=False,
                                                     num_workers=args.workers, pin_memory=is_gpu, sampler=None)
    dataset.threshset_loader = torch.utils.data.DataLoader(dataset.threshset, batch_size=args.batch_size, shuffle=False,
                                                           num_workers=args.workers, pin_memory=is_gpu, sampler=None)

    # Load open set datasets
    openset_datasets_names = args.openset_datasets.strip().split(',')
    openset_datasets = []
    for openset_dataset in openset_datasets_names:
        openset_data_init_method = getattr(datasets, openset_dataset)
        openset_datasets.append(openset_data_init_method(torch.cuda.is_available(), args))

    # Initialize empty model
    net_init_method = getattr(architectures, args.architecture)
    model = net_init_method(device, num_classes, num_colors, args).to(device)
    model = torch.nn.DataParallel(model).to(device)
    print(model)

    # load model (using the resume functionality)
    assert(os.path.isfile(args.resume)), "=> no model checkpoint found at '{}'".format(args.resume)

    # Fill the random model with the parameters of the checkpoint
    print("=> loading checkpoint '{}'".format(args.resume))
    checkpoint = torch.load(args.resume)
    best_prec = checkpoint['best_prec']
    best_loss = checkpoint['best_loss']
    print("Saved model's validation accuracy: ", best_prec)
    print("Saved model's validation loss: ", best_loss)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # set the save path to the directory from which the model has been loaded
    save_path = os.path.dirname(args.resume)

    # start of the model evaluation on the training dataset and fitting
    print("Evaluating original train dataset: " + args.dataset + ". This may take a while...")
    dataset_eval_dict_train = eval_dataset(model, dataset.train_loader, num_classes, device,
                                           latent_var_samples=args.var_samples, model_var_samples=args.model_samples)
    print("Training accuracy: ", dataset_eval_dict_train["accuracy"])

    # Get the mean of z for correctly classified data inputs
    mean_zs = get_means(dataset_eval_dict_train["zs_correct"])

    # visualize the mean z vectors
    mean_zs_tensor = torch.stack(mean_zs, dim=0)
    visualize_means(mean_zs_tensor, num_classes, args.dataset, save_path, "z")

    # calculate each correctly classified example's distance to the mean z
    distances_to_z_means_correct_train = calc_distances_to_means(mean_zs, dataset_eval_dict_train["zs_correct"],
                                                                 args.distance_function)

    # Weibull fitting
    # set tailsize according to command line parameters (according to percentage of dataset size)
    tailsize = int(len(dataset.trainset) * args.openset_weibull_tailsize / num_classes)
    print("Fitting Weibull models with tailsize: " + str(tailsize))
    tailsizes = [tailsize] * num_classes
    weibull_models, valid_weibull = fit_weibull_models(distances_to_z_means_correct_train, tailsizes)
    assert valid_weibull, "Weibull fit is not valid"

    # ------------------------------------------------------------------------------------------
    # Fitting on train dataset complete. Determine rejection thresholds/priors on the created split set
    # ------------------------------------------------------------------------------------------
    print("Evaluating original threshold split dataset: " + args.dataset + ". This may take a while...")
    threshset_eval_dict = eval_dataset(model, dataset.threshset_loader, num_classes, device,
                                       latent_var_samples=args.var_samples, model_var_samples=args.model_samples)

    # Again calculate distances to mean z
    print("Split set accuracy: ", threshset_eval_dict["accuracy"])
    distances_to_z_means_threshset = calc_distances_to_means(mean_zs, threshset_eval_dict["zs_correct"],
                                                             args.distance_function)
    # get Weibull outlier probabilities for thresh set
    outlier_probs_threshset = calc_outlier_probs(weibull_models, distances_to_z_means_threshset)
    threshset_classification = calc_openset_classification(outlier_probs_threshset, num_classes,
                                                           num_outlier_threshs=100)
    # also check outlier detection based on entropy
    max_entropy = np.max(threshset_eval_dict["out_entropy"])
    threshset_entropy_classification = calc_entropy_classification(threshset_eval_dict["out_entropy"],
                                                                   max_entropy,
                                                                   num_outlier_threshs=100)

    # determine rejection priors based on 5% of the split data considered as inlying
    if (np.array(threshset_classification["outlier_percentage"]) <= 0.05).any() == True:
        EVT_prior_index = np.argwhere(np.array(threshset_classification["outlier_percentage"])
                                      <= 0.05)[0][0]
        EVT_prior = threshset_classification["thresholds"][EVT_prior_index]
    else:
        EVT_prior = 0.5
        EVT_prior_index = 50

    if (np.array(threshset_entropy_classification["entropy_outlier_percentage"]) <= 0.05).any() == True:
        entropy_threshold_index = np.argwhere(np.array(threshset_entropy_classification["entropy_outlier_percentage"])
                                              <= 0.05)[0][0]
        entropy_threshold = threshset_entropy_classification["entropy_thresholds"][entropy_threshold_index]
    else:
        # this should never actually happen
        entropy_threshold = np.median(threshset_entropy_classification["entropy_thresholds"])
        entropy_threshold_index = 50

    print("EVT prior: " + str(EVT_prior) + "; Entropy threshold: " + str(entropy_threshold))

    # ------------------------------------------------------------------------------------------
    # Beginning of all testing/open set recognition on test and unknown sets.
    # ------------------------------------------------------------------------------------------
    # We evaluate the validation set to later evaluate trained dataset's statistical inlier/outlier estimates.
    print("Evaluating original validation dataset: " + args.dataset + ". This may take a while...")
    dataset_eval_dict = eval_dataset(model, dataset.val_loader, num_classes, device,
                                     latent_var_samples=args.var_samples, model_var_samples=args.model_samples)

    # Again calculate distances to mean z
    print("Validation accuracy: ", dataset_eval_dict["accuracy"])
    distances_to_z_means_correct = calc_distances_to_means(mean_zs, dataset_eval_dict["zs_correct"],
                                                           args.distance_function)

    # Evaluate outlier probability of trained dataset's validation set
    outlier_probs_correct = calc_outlier_probs(weibull_models, distances_to_z_means_correct)

    dataset_classification_correct = calc_openset_classification(outlier_probs_correct, num_classes,
                                                                 num_outlier_threshs=100)
    dataset_entropy_classification_correct = calc_entropy_classification(dataset_eval_dict["out_entropy"],
                                                                         max_entropy,
                                                                         num_outlier_threshs=100)

    print(args.dataset + '(trained) EVT outlier percentage: ' +
          str(dataset_classification_correct["outlier_percentage"][EVT_prior_index]))
    print(args.dataset + '(trained) entropy outlier percentage: ' +
          str(dataset_entropy_classification_correct["entropy_outlier_percentage"][entropy_threshold_index]))

    # Repeat process for open set recognition on unseen datasets (
    openset_dataset_eval_dicts = collections.OrderedDict()
    openset_outlier_probs_dict = collections.OrderedDict()
    openset_classification_dict = collections.OrderedDict()
    openset_entropy_classification_dict = collections.OrderedDict()

    for od, openset_dataset in enumerate(openset_datasets):
        print("Evaluating openset dataset: " + openset_datasets_names[od] + ". This may take a while...")
        openset_dataset_eval_dict = eval_openset_dataset(model, openset_dataset.val_loader, num_classes,
                                                         device, latent_var_samples=args.var_samples,
                                                         model_var_samples=args.model_samples)

        openset_distances_to_z_means = calc_distances_to_means(mean_zs, openset_dataset_eval_dict["zs"],
                                                               args.distance_function)

        openset_outlier_probs = calc_outlier_probs(weibull_models, openset_distances_to_z_means)

        # getting outlier classification accuracies across the entire datasets
        openset_classification = calc_openset_classification(openset_outlier_probs, num_classes,
                                                             num_outlier_threshs=100)

        openset_entropy_classification = calc_entropy_classification(openset_dataset_eval_dict["out_entropy"],
                                                                     max_entropy, num_outlier_threshs=100)

        # dictionary of dictionaries: per datasetname one dictionary with respective values
        openset_dataset_eval_dicts[openset_datasets_names[od]] = openset_dataset_eval_dict
        openset_outlier_probs_dict[openset_datasets_names[od]] = openset_outlier_probs
        openset_classification_dict[openset_datasets_names[od]] = openset_classification
        openset_entropy_classification_dict[openset_datasets_names[od]] = openset_entropy_classification

    # print outlier rejection values for all unseen unknown datasets
    for other_data_name, other_data_dict in openset_classification_dict.items():
        print(other_data_name + ' EVT outlier percentage: ' +
              str(other_data_dict["outlier_percentage"][entropy_threshold_index]))

    for other_data_name, other_data_dict in openset_entropy_classification_dict.items():
        print(other_data_name + ' entropy outlier percentage: ' +
              str(other_data_dict["entropy_outlier_percentage"][entropy_threshold_index]))

    # joint prediction uncertainty plot for all datasets
    if (args.train_var and args.var_samples > 1) or args.model_samples > 1:
        visualize_classification_uncertainty(dataset_eval_dict["out_mus_correct"],
                                             dataset_eval_dict["out_sigmas_correct"],
                                             openset_dataset_eval_dicts,
                                             "out_mus", "out_sigmas",
                                             args.dataset + ' (trained)',
                                             args.var_samples, save_path)

    # visualize the outlier probabilities
    visualize_weibull_outlier_probabilities(outlier_probs_correct, openset_outlier_probs_dict,
                                            args.dataset + ' (trained)', save_path, tailsize)

    visualize_classification_scores(dataset_eval_dict["out_mus_correct"], openset_dataset_eval_dicts, 'out_mus',
                                    args.dataset + ' (trained)', save_path)

    visualize_entropy_histogram(dataset_eval_dict["out_entropy"], openset_dataset_eval_dicts,
                                dataset_entropy_classification_correct["entropy_thresholds"][-1], "out_entropy",
                                args.dataset + ' (trained)', save_path)

    # joint plot for outlier detection accuracy for seen and both unseen datasets
    visualize_openset_classification(dataset_classification_correct["outlier_percentage"],
                                     openset_classification_dict, "outlier_percentage",
                                     args.dataset + ' (trained)',
                                     dataset_classification_correct["thresholds"], save_path, tailsize)

    visualize_entropy_classification(dataset_entropy_classification_correct["entropy_outlier_percentage"],
                                     openset_entropy_classification_dict, "entropy_outlier_percentage",
                                     args.dataset + ' (trained)',
                                     dataset_entropy_classification_correct["entropy_thresholds"], save_path)


if __name__ == '__main__':
    main()
