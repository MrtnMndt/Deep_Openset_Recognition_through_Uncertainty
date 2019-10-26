import torch
import numpy as np
import libmr


def get_means(tensors_list):
    """
    Calculate the mean of a list of tensors for each tensor in the list. In our case the list typically contains
    a tensor for each class, such as the per class z values.

    Parameters:
        tensors_list (list): List of Tensors

    Returns:
        list: List of Tensors containing mean vectors
    """

    means = []
    for i in range(len(tensors_list)):
        if isinstance(tensors_list[i], torch.Tensor):
            means.append(torch.mean(tensors_list[i], dim=0))
        else:
            means.append([])

    return means


def calc_distances_to_means(means, tensors, distance_function='cosine'):
    """
    Function to calculate distances between tensors, in our case the mean zs per class and z for each input.
    Wrapper around torch.nn.functonal distances with specification of which distance function to choose.

    Parameters:
        means (list): List of length corresponding to number of classes containing torch tensors (typically mean zs).
        tensors (list): List of length corresponding to number of classes containing tensors (typically zs).
        distance_function (str): Specification of distance function. Choice of cosine|euclidean|mix.

    Returns:
        list: List of length corresponding to number of classes containing tensors with distance values
    """

    def distance_func(a, b, w_eucl, w_cos):
        """
        Weighted distance function consisting of cosine and euclidean components.

        Parameters:
            a (torch.Tensor): First tensor.
            b (torch.Tensor): Second tensor.
            w_eucl (float): Weight for the euclidean distance term.
            w_cos (float): Weight for the cosine similarity term.
        """
        d = w_cos * (1 - torch.nn.functional.cosine_similarity(a.view(1, -1), b)) + \
            w_eucl * torch.nn.functional.pairwise_distance(a.view(1, -1), b, p=2)
        return d

    distances = []

    # weight values for individual distance components
    w_eucl = 0.0
    w_cos = 0.0
    if distance_function == 'euclidean':
        w_eucl = 1.0
    elif distance_function == 'cosine':
        w_cos = 1.0
    elif distance_function == 'mix':
        w_eucl = 0.5
        w_cos = 0.5
    else:
        raise ValueError("distance function not implemented")

    # loop through each class in means and calculate the distances with the respective tensor.
    for i in range(len(means)):
        # check for tensor type, e.g. list could be empty
        if isinstance(tensors[i], torch.Tensor) and isinstance(means[i], torch.Tensor):
            distances.append(distance_func(means[i], tensors[i], w_eucl, w_cos))
        else:
            distances.append([])

    return distances


def fit_weibull_models(distribution_values, tailsizes, num_max_fits=50):
    """
    Function to fit weibull models on distribution values per class. The distribution values in our case are the
    distances of an inputs approximate posterior value to the per class mean latent z, i.e. The Weibull model fits
    regions of high density and gives credible intervals.
    The tailsize specifies how many outliers are expected in the dataset for which the model has been trained.
    We use libmr https://github.com/Vastlab/libMR (installable through e.g. pip) for the Weibull model fitting.

    Parameters:
        distribution_values (list): Values on which the fit is conducted. In our case latent space distances.
        tailsizes (list): List of integers, specifying tailsizes per class. For a balanced dataset typically the same.
        num_max_fits (int): Number of attempts to fit the Weibull models before timing out and returning unsuccessfully.

    Returns:
        list: List of Weibull models with their respective parameters (stored in libmr class instances).
    """

    weibull_models = []

    # loop through the list containing distance values per class
    for i in range(len(distribution_values)):
        # for each class set the initial success to False and number of attempts to 0
        is_valid = False
        count = 0

        # If the list contains distance values conduct a fit. If it is empty, e.g. because there is not a single
        # prediction for the corresponding class, continue with the next class. Note that the latter isn't expected for
        # a model that has been trained for even just a short while.
        if isinstance(distribution_values[i], torch.Tensor):
            distribution_values[i] = distribution_values[i].cpu().numpy()
            # weibull model per class
            weibull_models.append(libmr.MR())
            # attempt num_max_fits many fits before aborting
            while is_valid is False and count < num_max_fits:
                # conduct the fit with libmr
                weibull_models[i].fit_high(distribution_values[i], tailsizes[i])
                is_valid = weibull_models[i].is_valid
                count += 1
            if not is_valid:
                print("Weibull fit for class " + str(i) + " not successful after " + str(num_max_fits) + " attempts")
                return weibull_models, False
        else:
            weibull_models.append([])

    return weibull_models, True


def calc_outlier_probs(weibull_models, distances):
    """
    Calculates statistical outlier probability using the weibull models' CDF.

    Note that we have coded this function to loop over each class because we have previously categorized the distances
    into their respective classes already.

    Parameters:
        weibull_models (list): List of libmr class instances containing the Weibull model parameters and functions.
        distances (list): List of per class torch tensors or numpy arrays with latent space distance values.

    Returns:
        list: List of length corresponding to number of classes with outlier probabilities for each respective input.
    """

    outlier_probs = []
    # loop through all classes, i.e. all available weibull models as there is one weibull model per class.
    for i in range(len(weibull_models)):
        # optionally convert the type of the distance vectors
        if isinstance(distances[i], torch.Tensor):
            distances[i] = distances[i].cpu().numpy().astype(np.double)
        elif isinstance(distances[i], list):
            # empty list
            outlier_probs.append([])
            continue
        else:
            distances[i] = distances[i].astype(np.double)

        # use the Weibull models' CDF to evaluate statistical outlier rejection probabilities.
        outlier_probs.append(weibull_models[i].w_score_vector(distances[i]))

    return outlier_probs


def calc_openset_classification(data_outlier_probs, num_classes, num_outlier_threshs=50):
    """
    Calculates the percentage of dataset outliers given a set of outlier probabilities over a range of rejection priors.

    Parameters:
         data_outlier_probs (list): List of outlier probabilities for an entire dataset, categorized by class.
         num_classes (int): Number of classes.
         num_outlier_threshs (int): Number of outlier rejection priors (evenly spread over the interval (0,1)).

    Returns:
        dict: Dictionary containing outlier percentages and corresponding rejection prior values.
    """

    dataset_outliers = []
    threshs = []

    # loop through each rejection prior value and evaluate the percentage of the dataset being considered as
    # statistical outliers, i.e. each data point's outlier probability > rejection prior.
    for i in range(num_outlier_threshs - 1):
        outlier_threshold = (i + 1) * (1.0 / num_outlier_threshs)
        threshs.append(outlier_threshold)

        dataset_outliers.append(0)
        total_dataset = 0

        for j in range(num_classes):
            total_dataset += len(data_outlier_probs[j])

            for k in range(len(data_outlier_probs[j])):
                if data_outlier_probs[j][k] > outlier_threshold:
                    dataset_outliers[i] += 1

        dataset_outliers[i] = dataset_outliers[i] / float(total_dataset)

    return {"thresholds": threshs, "outlier_percentage": dataset_outliers}


def calc_entropy_classification(dataset_entropies, max_thresh_value, num_outlier_threshs=50):
    """
    Calculates the percentage of dataset outliers given a set of entropies over a range of rejection priors.
    Parameters:
         dataset_entropies (list): List of entropies for the entire dataset (each instance)
         num_outlier_threshs (int): Number of outlier rejection priors (evenly spread over the interval (0,1)).
    Returns:
        dict: Dictionary containing outlier percentages and corresponding rejection prior values.
    """

    dataset_outliers = []
    threshs = []

    total_dataset = float(len(dataset_entropies))

    # loop through each rejection prior value and evaluate the percentage of the dataset being considered as
    # statistical outliers, i.e. each data point's outlier probability > rejection prior.
    for i in range(num_outlier_threshs - 1):
        outlier_threshold = (i + 1) * (max_thresh_value / num_outlier_threshs)
        threshs.append(outlier_threshold)

        dataset_outliers.append(0)

        for k in range(len(dataset_entropies)):
            if dataset_entropies[k] > outlier_threshold:
                dataset_outliers[i] += 1

        dataset_outliers[i] = dataset_outliers[i] / total_dataset

    return {"entropy_thresholds": threshs, "entropy_outlier_percentage": dataset_outliers}
