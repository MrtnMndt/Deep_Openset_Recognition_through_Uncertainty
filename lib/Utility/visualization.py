import torch
import os
import seaborn as sns
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# matplotlib backend, required for plotting of images to tensorboard
matplotlib.use('Agg')

# setting font sizes
title_font_size = 60
axes_font_size = 45
legend_font_size = 39
ticks_font_size = 48

# setting seaborn specifics
sns.set(font_scale=2.5)
sns.set_style("whitegrid")
colors = sns.color_palette("Set2")
pal = sns.cubehelix_palette(10, light=0.0)
linestyles = [(0, (1, 3)),  # 'dotted'
              (0, (1, 1)),  # 'densely dotted'
              (0, (2, 2)),  # 'dashed'
              (0, (3, 1)),  # 'densely dashed'
              (0, (3, 3, 1, 3)),  # 'dashdotted'
              (0, (3, 1, 1, 1)),  # 'densely dashdotted'
              (0, (3, 3, 1, 3, 1, 3)),  # 'dashdotdotted'
              (0, (3, 1, 1, 1, 1, 1))]  # 'densely dashdotdotted'


def args_to_tensorboard(writer, args):
    """
    Takes command line parser arguments and formats them to
    display them in TensorBoard text.

    Parameters:
        writer (tensorboard.SummaryWriter): TensorBoard SummaryWriter instance.
        args (dict): dictionary of command line arguments
    """

    txt = ""
    for arg in vars(args):
        txt += arg + ": " + str(getattr(args, arg)) + "<br/>"

    writer.add_text('command_line_parameters', txt, 0)


def visualize_means(means, num_classes, data_name, save_path, name):
    """
    Visualization of means, e.g. of latent code z.

    Parameters:
        means (torch.Tensor): 2-D Tensor with one mean z vector per class.
        num_classes (int): Defines number of classes.
        data_name (str): Dataset name. Used for naming.
        save_path (str): Saving path.
        name (str): Name for type of mean, e.g. "z".
    """

    classes = np.arange(0, num_classes)

    plt.figure(figsize=(20, 20))
    ax = sns.heatmap(means.cpu().numpy(), cmap="BrBG")
    ax.set_title(data_name, fontsize=title_font_size)
    ax.set_xlabel(name + ' mean activations', fontsize=axes_font_size)
    ax.set_yticklabels(classes, rotation=0)
    plt.savefig(os.path.join(save_path, name + '_mean_activations.png'), bbox_inches='tight')


def visualize_classification_uncertainty(data_mus, data_sigmas, other_data_dicts, other_data_mu_key,
                                         other_data_sigma_key, data_name, num_samples, save_path):
    """
    Visualization of prediction uncertainty computed over multiple samples for each input.

    Parameters:
        data_mus (list or torch.Tensor): Encoded mu values for trained dataset's validation set.
        data_sigmas (list or torch.Tensor): Encoded sigma values for trained dataset's validation set.
        other_data_dicts (dictionary of dictionaries): A dataset with values per dictionary, among them mus and sigmas
        other_data_mu_key (str): Dictionary key for the mus
        other_data_sigma_key (str): Dictionary key for the sigmas
        data_name (str): Original dataset's name.
        num_samples (int): Number of used samples to obtain prediction values.
        save_path (str): Saving path.
    """

    data_mus = [y for x in data_mus for y in x]
    data_sigmas = [y for x in data_sigmas for y in x]

    plt.figure(figsize=(20, 14))
    plt.scatter(data_mus, data_sigmas, label=data_name, s=75, c=colors[0], alpha=1.0)

    c = 0
    for other_data_name, other_data_dict in other_data_dicts.items():
        other_data_mus = [y for x in other_data_dict[other_data_mu_key] for y in x]
        other_data_sigmas = [y for x in other_data_dict[other_data_sigma_key] for y in x]
        plt.scatter(other_data_mus, other_data_sigmas, label=other_data_name, s=75, c=colors[c], alpha=0.3,
                    marker='*')
        c += 1

    plt.xlabel("Prediction mean", fontsize=axes_font_size)
    plt.ylabel("Prediction standard deviation", fontsize=axes_font_size)
    plt.xlim(left=-0.05, right=1.05)
    plt.ylim(bottom=-0.05, top=0.55)
    plt.legend(loc=1, fontsize=legend_font_size)
    plt.savefig(os.path.join(save_path, data_name + '_vs_' + ",".join(list(other_data_dicts.keys())) +
                             '_classification_uncertainty_' + str(num_samples) + '_samples.pdf'),
                bbox_inches='tight')


def visualize_classification_scores(data, other_data_dicts, dict_key, data_name, save_path):
    """
    Visualization of classification scores per dataset.

    Parameters:
        data (list): Classification scores.
        other_data_dicts (dictionary of dictionaries): Dictionary of key-value pairs per dataset
        dict_key (string): Dictionary key to plot
        data_name (str): Original trained dataset's name.
        save_path (str): Saving path.
    """

    data = [y for x in data for y in x]

    plt.figure(figsize=(20, 20))
    plt.hist(data, label=data_name, alpha=1.0, bins=20, color=colors[0])

    c = 0
    for other_data_name, other_data_dict in other_data_dicts.items():
        other_data = [y for x in other_data_dict[dict_key] for y in x]
        plt.hist(other_data, label=other_data_name, alpha=0.5, bins=20, color=colors[c])
        c += 1

    plt.title("Dataset classification", fontsize=title_font_size)
    plt.xlabel("Classification confidence", fontsize=axes_font_size)
    plt.ylabel("Number of images", fontsize=axes_font_size)
    plt.legend(loc=0)
    plt.xlim(left=-0.0, right=1.05)

    plt.savefig(os.path.join(save_path, data_name + '_' + ",".join(list(other_data_dicts.keys()))
                             + '_classification_scores.png'),
                bbox_inches='tight')
                
                
def visualize_entropy_histogram(data, other_data_dicts, max_entropy, dict_key, data_name, save_path):
    """
    Visualization of the entropy the datasets.

    Parameters:
        data (list):
        other_data_dicts (dictionary of dictionaries): Dictionary of key-value pairs per dataset
        dict_key (str): Dictionary key to plot
        data_name (str): Original trained dataset's name.
        save_path (str): Saving path.
    """
    data = [x for x in data]

    plt.figure(figsize=(20, 20))
    plt.hist(data, label=data_name, alpha=1.0, bins=25, color=colors[0])

    c = 0
    for other_data_name, other_data_dict in other_data_dicts.items():
        other_data = [x for x in other_data_dict[dict_key]]
        plt.hist(other_data, label=other_data_name, alpha=0.5, bins=25, color=colors[c])
        c += 1

    plt.title("Dataset classification entropy", fontsize=title_font_size)
    plt.xlabel("Classification entropy", fontsize=axes_font_size)
    plt.ylabel("Number of images", fontsize=axes_font_size)
    plt.legend(loc=0)
    plt.xlim(left=-0.0, right=max_entropy)
    plt.savefig(os.path.join(save_path, data_name + '_' + ",".join(list(other_data_dicts.keys()))
                             + '_classification_entropies.png'),
                bbox_inches='tight')


def visualize_weibull_outlier_probabilities(data_outlier_probs, other_data_outlier_probs_dict,
                                            data_name, save_path, tailsize):
    """
    Visualization of Weibull CDF outlier probabilites.

    Parameters:
        data_outlier_probs (np.array): Outlier probabilities for each input of the trained dataset's validation set.
        other_data_outlier_probs_dict (dictionary): Outlier probabilities for each input of an unseen dataset.
        data_name (str): Original trained dataset's name.
        save_path (str): Saving path.
        tailsize (int): Fitted Weibull model's tailsize.
    """

    data_outlier_probs = np.concatenate(data_outlier_probs, axis=0)

    data_weights = np.ones_like(data_outlier_probs) / float(len(data_outlier_probs))

    plt.figure(figsize=(20, 20))
    plt.hist(data_outlier_probs, label=data_name, weights=data_weights, bins=50, color=colors[0],
             alpha=1.0, edgecolor='white', linewidth=5)

    c = 0
    for other_data_name, other_data_outlier_probs in other_data_outlier_probs_dict.items():
        other_data_outlier_probs = np.concatenate(other_data_outlier_probs, axis=0)
        other_data_weights = np.ones_like(other_data_outlier_probs) / float(len(other_data_outlier_probs))
        plt.hist(other_data_outlier_probs, label=other_data_name, weights=other_data_weights,
             bins=50, color=colors[c], alpha=0.5, edgecolor='white', linewidth=5)
        c += 1

    plt.title("Outlier probabilities: tailsize " + str(tailsize), fontsize=title_font_size)
    plt.xlabel("Outlier probability according to Weibull CDF", fontsize=axes_font_size)
    plt.ylabel("Percentage", fontsize=axes_font_size)
    plt.xlim(left=-0.05, right=1.05)
    plt.ylim(bottom=-0.05, top=1.05)
    plt.legend(loc=0)

    plt.savefig(os.path.join(save_path, data_name + '_' + ",".join(list(other_data_outlier_probs_dict.keys()))
                             + '_weibull_outlier_probabilities_tailsize_'
                             + str(tailsize) + '.png'), bbox_inches='tight')


def visualize_openset_classification(data, other_data_dicts, dict_key, data_name,
                                     thresholds, save_path, tailsize):
    """
    Visualization of percentage of datasets considered as statistical outliers evaluated for different
    Weibull CDF rejection priors.

    Parameters:
        data (list): Dataset outlier percentages per rejection prior value for the trained dataset's validation set.
        other_data_dicts (dictionary of dictionaries):
            Dataset outlier percentages per rejection prior value for an unseen dataset.
        dict_key (str): Dictionary key of the values to visualize
        data_name (str): Original trained dataset's name.
        thresholds (list): List of integers with rejection prior values.
        save_path (str): Saving path.
        tailsize (int): Weibull model's tailsize.
    """

    lw = 10
    plt.figure(figsize=(20, 20))
    plt.plot(thresholds, data, label=data_name, color=colors[0], linestyle='solid', linewidth=lw)

    c = 0
    for other_data_name, other_data_dict in other_data_dicts.items():
        plt.plot(thresholds, other_data_dict[dict_key], label=other_data_name, color=colors[c],
                 linestyle=linestyles[c % len(linestyles)], linewidth=lw)
        c += 1

    plt.xlabel(r"Weibull CDF outlier rejection prior $\Omega_t$", fontsize=axes_font_size)
    plt.ylabel("Percentage of dataset outliers", fontsize=axes_font_size)
    plt.xlim(left=-0.05, right=1.05)
    plt.ylim(bottom=-0.05, top=1.05)
    plt.legend(loc=0, fontsize=legend_font_size - 15)
    plt.savefig(os.path.join(save_path, data_name + '_' + ",".join(list(other_data_dicts.keys())) +
                             '_outlier_classification' + '_tailsize_' + str(tailsize) + '.pdf'),
                bbox_inches='tight')


def visualize_entropy_classification(data, other_data_dicts, dict_key, data_name,
                                     thresholds, save_path):
    """
    Visualization of percentage of datasets considered as statistical outliers evaluated for different
    entropy thresholds.

    Parameters:
        data (list): Dataset outlier percentages per rejection prior value for the trained dataset's validation set.
        other_data_dicts (dictionary of dictionaries):
            Dataset outlier percentages per rejection prior value for an unseen dataset.
        dict_key (str): Dictionary key of the values to visualize
        data_name (str): Original trained dataset's name.
        thresholds (list): List of integers with rejection prior values.
        save_path (str): Saving path.
    """

    lw = 10
    plt.figure(figsize=(20, 20))
    plt.plot(thresholds, data, label=data_name, color=colors[0], linestyle='solid', linewidth=lw)

    c = 0
    for other_data_name, other_data_dict in other_data_dicts.items():
        plt.plot(thresholds, other_data_dict[dict_key], label=other_data_name, color=colors[c],
                 linestyle=linestyles[c % len(linestyles)], linewidth=lw)
        c += 1

    plt.xlabel(r"Dataset entropy", fontsize=axes_font_size)
    plt.ylabel("Percentage of dataset outliers", fontsize=axes_font_size)
    plt.xlim(left=-0.05, right=thresholds[-1])
    plt.ylim(bottom=-0.05, top=1.05)
    plt.legend(loc=0, fontsize=legend_font_size - 15)
    plt.savefig(os.path.join(save_path, data_name + '_' + ",".join(list(other_data_dicts.keys())) +
                             '_entropy_outlier_classification' + '.pdf'),
                bbox_inches='tight')
