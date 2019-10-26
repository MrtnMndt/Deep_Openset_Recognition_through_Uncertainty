import torch
import numpy as np


def eval_var_dataset(model, data_loader, num_classes, device, latent_var_samples=1, model_var_samples=1):
    """
    Evaluates an entire dataset with the variational or joint model and stores z values, latent mus and sigmas and
    output predictions according to whether the classification is correct or not.
    The values for correct predictions can later be used for plotting or fitting of Weibull models.

    Parameters:
        model (torch.nn.module): Trained model.
        data_loader (torch.utils.data.DataLoader): The dataset loader.
        num_classes (int): Number of classes.
        device (str): Device to compute on.
        latent_var_samples (int): Number of latent space variational samples.
        model_var_samples (int): Number of variational samples of the entire model, e.g. MC dropout.

    Returns:
        dict: Dictionary of results and latent values, separated by whether the classification was correct or not.
    """

    # switch to evaluation mode unless MC dropout is active
    if model_var_samples > 1:
        model.train()
    else:
        model.eval()

    correctly_identified = 0
    tot_samples = 0

    out_mus_correct = []
    out_sigmas_correct = []
    out_mus_false = []
    out_sigmas_false = []
    encoded_mus_correct = []
    encoded_mus_false = []
    encoded_sigmas_correct = []
    encoded_sigmas_false = []
    zs_correct = []
    zs_false = []

    out_entropy = []
   
    for i in range(num_classes):
        out_mus_correct.append([])
        out_mus_false.append([])
        out_sigmas_correct.append([])
        out_sigmas_false.append([])
        encoded_mus_correct.append([])
        encoded_mus_false.append([])
        encoded_sigmas_correct.append([])
        encoded_sigmas_false.append([])
        zs_false.append([])
        zs_correct.append([])
        
    # evaluate the encoder and classifier and store results in corresponding lists according to predicted class.
    # Prediction mean confidence and uncertainty is also obtained if amount of latent samples is greater than one.
    with torch.no_grad():
        for j, (inputs, classes) in enumerate(data_loader):
            inputs, classes = inputs.to(device), classes.to(device)

            out_samples = torch.zeros(model_var_samples, latent_var_samples, inputs.size(0), num_classes).to(device)
            z_samples = torch.zeros(model_var_samples, latent_var_samples,
                                    inputs.size(0), model.module.latent_dim).to(device)

            # sampling the model, then z and classifying
            for k in range(model_var_samples):
                encoded_mu, encoded_std = model.module.encode(inputs)

                for i in range(latent_var_samples):
                    z = model.module.reparameterize(encoded_mu, encoded_std)
                    z_samples[k][i] = z

                    cl = model.module.classifier(z)
                    out = torch.nn.functional.softmax(cl, dim=1)
                    out_samples[k][i] = out

            out_mean = torch.mean(torch.mean(out_samples, dim=0), dim=0)
            if model_var_samples > 1:
                out_std = torch.std(torch.mean(out_samples, dim=0), dim=0)
            else:
                out_std = torch.squeeze(torch.std(out_samples, dim=1))

            zs_mean = torch.mean(torch.mean(z_samples, dim=0), dim=0)
            
            # calculate entropy for the means of samples: - sum pc*log(pc)
            eps = 1e-10
            out_entropy.append(-torch.sum(out_mean*torch.log(out_mean + eps), dim=1).cpu().detach().numpy())

            # for each input and respective prediction store independently depending on whether classification was
            # correct. The list of correct classifications is later used for fitting of Weibull models if the
            # data_loader is loading the training set.
            for i in range(inputs.size(0)):
                tot_samples += 1
                idx = torch.argmax(out_mean[i]).item()
                if classes[i].item() != idx:
                    out_mus_false[idx].append(out_mean[i][idx].item())
                    out_sigmas_false[idx].append(out_std[i][idx].item())
                    encoded_mus_false[idx].append(encoded_mu[i].data)
                    encoded_sigmas_false[idx].append(encoded_std[i].data)
                    zs_false[idx].append(zs_mean[i].data)
                else:
                    correctly_identified += 1
                    out_mus_correct[idx].append(out_mean[i][idx].item())
                    out_sigmas_correct[idx].append(out_std[i][idx].item())
                    encoded_mus_correct[idx].append(encoded_mu[i].data)
                    encoded_sigmas_correct[idx].append(encoded_std[i].data)
                    zs_correct[idx].append(zs_mean[i].data)

    acc = correctly_identified / float(tot_samples)

    # stack list of tensors into tensors
    for i in range(len(encoded_mus_correct)):
        if len(encoded_mus_correct[i]) > 0:
            encoded_mus_correct[i] = torch.stack(encoded_mus_correct[i], dim=0)
            encoded_sigmas_correct[i] = torch.stack(encoded_sigmas_correct[i], dim=0)
            zs_correct[i] = torch.stack(zs_correct[i], dim=0)
        if len(encoded_mus_false[i]) > 0:
            encoded_mus_false[i] = torch.stack(encoded_mus_false[i], dim=0)
            encoded_sigmas_false[i] = torch.stack(encoded_sigmas_false[i], dim=0)
            zs_false[i] = torch.stack(zs_false[i], dim=0)

    out_entropy = np.concatenate(out_entropy).ravel().tolist()

    # Return a dictionary containing all the stored values
    return {"accuracy": acc, "encoded_mus_correct": encoded_mus_correct, "encoded_mus_false": encoded_mus_false,
            "encoded_sigmas_correct": encoded_sigmas_correct, "encoded_sigmas_false": encoded_sigmas_false,
            "zs_correct": zs_correct, "zs_false": zs_false,
            "out_mus_correct": out_mus_correct, "out_sigmas_correct": out_sigmas_correct,
            "out_mus_false": out_mus_false, "out_sigmas_false": out_sigmas_false,
            "out_entropy": out_entropy}


def eval_dataset(model, data_loader, num_classes, device, latent_var_samples=1, model_var_samples=1):
    """
    Evaluates an entire dataset with the model and stores z values and
    output predictions according to whether the classification is correct or not.
    The values for correct predictions can later be used for plotting or fitting of Weibull models.

    Parameters:
        model (torch.nn.module): Trained model.
        data_loader (torch.utils.data.DataLoader): The dataset loader.
        num_classes (int): Number of classes.
        device (str): Device to compute on.
        model_var_samples (int): Number of variational samples of the entire model, e.g. MC dropout.

    Returns:
        dict: Dictionary of results and latent values, separated by whether the classification was correct or not.
    """

    # switch to evaluation mode unless MC dropout is active
    if model_var_samples > 1:
        model.train()
    else:
        model.eval()

    correctly_identified = 0
    tot_samples = 0

    out_mus_correct = []
    out_sigmas_correct = []
    out_mus_false = []
    out_sigmas_false = []
    zs_correct = []
    zs_false = []

    out_entropy = []

    for i in range(num_classes):
        out_mus_correct.append([])
        out_mus_false.append([])
        out_sigmas_correct.append([])
        out_sigmas_false.append([])
        zs_false.append([])
        zs_correct.append([])

    # evaluate the encoder and classifier and store results in corresponding lists according to predicted class.
    # Prediction mean confidence and uncertainty is also obtained if amount of latent samples is greater than one.
    with torch.no_grad():
        for j, (inputs, classes) in enumerate(data_loader):
            inputs, classes = inputs.to(device), classes.to(device)

            out_samples = torch.zeros(model_var_samples, inputs.size(0), num_classes).to(device)
            z_samples = torch.zeros(model_var_samples, inputs.size(0), model.module.latent_dim).to(device)

            # sampling the model, then z and classifying
            for k in range(model_var_samples):
                enc = model.module.encode(inputs)
                z = enc.view(enc.size(0), -1)
                z_samples[k] = z
                cl = model.module.classifier(z)
                out = torch.nn.functional.softmax(cl, dim=1)
                out_samples[k] = out

            out_mean = torch.mean(out_samples, dim=0)
            if model_var_samples > 1:
                out_std = torch.std(out_samples, dim=0)
            else:
                out_std = torch.squeeze(out_samples)

            zs_mean = torch.mean(z_samples, dim=0)

            # calculate entropy for the means of samples: - sum pc*log(pc)
            eps = 1e-10
            out_entropy.append(- torch.sum(out_mean * torch.log(out_mean + eps), dim=1).cpu().detach().numpy())

            # for each input and respective prediction store independently depending on whether classification was
            # correct. The list of correct classifications is later used for fitting of Weibull models if the
            # data_loader is loading the training set.
            for i in range(inputs.size(0)):
                tot_samples += 1
                idx = torch.argmax(out_mean[i]).item()
                if classes[i].item() != idx:
                    out_mus_false[idx].append(out_mean[i][idx].item())
                    out_sigmas_false[idx].append(out_std[i][idx].item())
                    zs_false[idx].append(zs_mean[i].data)
                else:
                    correctly_identified += 1
                    out_mus_correct[idx].append(out_mean[i][idx].item())
                    out_sigmas_correct[idx].append(out_std[i][idx].item())
                    zs_correct[idx].append(zs_mean[i].data)

    acc = correctly_identified / float(tot_samples)

    # stack list of tensors into tensors
    for i in range(len(zs_correct)):
        if len(zs_correct[i]) > 0:
            zs_correct[i] = torch.stack(zs_correct[i], dim=0)
        if len(zs_false[i]) > 0:
            zs_false[i] = torch.stack(zs_false[i], dim=0)

    out_entropy = np.concatenate(out_entropy).ravel().tolist()

    # Return a dictionary containing all the stored values
    return {"accuracy": acc, "zs_correct": zs_correct, "zs_false": zs_false,
            "out_mus_correct": out_mus_correct, "out_sigmas_correct": out_sigmas_correct,
            "out_mus_false": out_mus_false, "out_sigmas_false": out_sigmas_false, "out_entropy": out_entropy}


def eval_var_openset_dataset(model, data_loader, num_classes, device, latent_var_samples=1, model_var_samples=1):
    """
    Evaluates an entire dataset with the variational or joint model and stores z values, latent mus and sigmas and
    output predictions such that they can later be used for statistical outlier evaluation with the fitted Weibull
    models. This is merely for convenience to keep the rest of the code API the same. Note that the Weibull model's
    prediction of whether a sample from an unknown dataset is a statistical outlier or not can be done on an instance
    level. Similar to the eval_dataset function but without splitting of correct vs. false predictions as the dataset
    is unknown in the open-set scenario.

    Parameters:
        model (torch.nn.module): Trained model.
        data_loader (torch.utils.data.DataLoader): The dataset loader.
        num_classes (int): Number of classes.
        device (str): Device to compute on.
        latent_var_samples (int): Number of latent space variational samples.
        model_var_samples (int): Number of variational samples of the entire model, e.g. MC dropout.

    Returns:
        dict: Dictionary of results and latent values.
    """

    # switch to evaluation mode unless MC dropout is active
    if model_var_samples > 1:
        model.train()
    else:
        model.eval()

    out_mus = []
    out_sigmas = []
    encoded_mus = []
    encoded_sigmas = []
    zs = []

    out_entropy = []

    for i in range(num_classes):
        out_mus.append([])
        out_sigmas.append([])
        encoded_mus.append([])
        encoded_sigmas.append([])
        zs.append([])

    # evaluate the encoder and classifier and store results in corresponding lists according to predicted class.
    # Prediction mean confidence and uncertainty is also obtained if amount of latent samples is greater than one.
    with torch.no_grad():
        for j, (inputs, classes) in enumerate(data_loader):
            inputs, classes = inputs.to(device), classes.to(device)

            out_samples = torch.zeros(model_var_samples, latent_var_samples, inputs.size(0), num_classes).to(device)
            z_samples = torch.zeros(model_var_samples, latent_var_samples,
                                    inputs.size(0), model.module.latent_dim).to(device)

            # sampling the model, then z and classifying
            for k in range(model_var_samples):
                encoded_mu, encoded_std = model.module.encode(inputs)

                for i in range(latent_var_samples):
                    z = model.module.reparameterize(encoded_mu, encoded_std)
                    z_samples[k][i] = z

                    cl = model.module.classifier(z)
                    out = torch.nn.functional.softmax(cl, dim=1)
                    out_samples[k][i] = out

            # calculate the mean and std. Only removes a dummy dimension if number of variational samples is set to one.
            out_mean = torch.mean(torch.mean(out_samples, dim=0), dim=0)
            if model_var_samples > 1:
                out_std = torch.std(torch.mean(out_samples, dim=0), dim=0)
            else:
                out_std = torch.squeeze(torch.std(out_samples, dim=1))
            zs_mean = torch.mean(torch.mean(z_samples, dim=0), dim=0)
            
            # calculate entropy for the means of samples: - sum pc*log(pc)
            eps = 1e-10
            out_entropy.append(- torch.sum(out_mean*torch.log(out_mean + eps), dim=1).cpu().detach().numpy())

            # In contrast to the eval_dataset function, there is no split into correct or false values as the dataset
            # is unknown.
            for i in range(inputs.size(0)):
                idx = torch.argmax(out_mean[i]).item()
                out_mus[idx].append(out_mean[i][idx].item())
                out_sigmas[idx].append(out_std[i][idx].item())
                encoded_mus[idx].append(encoded_mu[i].data)
                encoded_sigmas[idx].append(encoded_std[i].data)
                zs[idx].append(zs_mean[i].data)

    # stack latent activations into a tensor
    for i in range(len(encoded_mus)):
        if len(encoded_mus[i]) > 0:
            encoded_mus[i] = torch.stack(encoded_mus[i], dim=0)
            encoded_sigmas[i] = torch.stack(encoded_sigmas[i], dim=0)
            zs[i] = torch.stack(zs[i], dim=0)

    out_entropy = np.concatenate(out_entropy).ravel().tolist()

    # Return a dictionary of stored values.
    return {"encoded_mus": encoded_mus, "encoded_sigmas": encoded_sigmas,
            "out_mus": out_mus, "out_sigmas": out_sigmas, "zs": zs,
            "out_entropy": out_entropy}


def eval_openset_dataset(model, data_loader, num_classes, device, latent_var_samples=1, model_var_samples=1):
    """
    Evaluates an entire dataset with the model and stores z values and
    output predictions such that they can later be used for statistical outlier evaluation with the fitted Weibull
    models. This is merely for convenience to keep the rest of the code API the same. Note that the Weibull model's
    prediction of whether a sample from an unknown dataset is a statistical outlier or not can be done on an instance
    level. Similar to the eval_dataset function but without splitting of correct vs. false predictions as the dataset
    is unknown in the open-set scenario.

    Parameters:
        model (torch.nn.module): Trained model.
        data_loader (torch.utils.data.DataLoader): The dataset loader.
        num_classes (int): Number of classes.
        device (str): Device to compute on.
        model_var_samples (int): Number of variational samples of the entire model, e.g. MC dropout.

    Returns:
        dict: Dictionary of results and latent values.
    """

    # switch to evaluation mode unless MC dropout is active
    if model_var_samples > 1:
        model.train()
    else:
        model.eval()

    out_mus = []
    out_sigmas = []
    zs = []

    out_entropy = []

    for i in range(num_classes):
        out_mus.append([])
        out_sigmas.append([])
        zs.append([])

    # evaluate the encoder and classifier and store results in corresponding lists according to predicted class.
    # Prediction mean confidence and uncertainty is also obtained if amount of latent samples is greater than one.
    with torch.no_grad():
        for j, (inputs, classes) in enumerate(data_loader):
            inputs, classes = inputs.to(device), classes.to(device)

            out_samples = torch.zeros(model_var_samples, inputs.size(0), num_classes).to(device)
            z_samples = torch.zeros(model_var_samples, inputs.size(0), model.module.latent_dim).to(device)

            # sampling the model, then z and classifying
            for k in range(model_var_samples):
                enc = model.module.encode(inputs)
                z = enc.view(enc.size(0), -1)
                z_samples[k] = z
                cl = model.module.classifier(enc)
                out = torch.nn.functional.softmax(cl, dim=1)
                out_samples[k] = out

            # calculate the mean and std. Only removes a dummy dimension if number of samples is set to one.
            out_mean = torch.mean(out_samples, dim=0)
            if model_var_samples > 1:
                out_std = torch.std(out_samples, dim=0)
            else:
                out_std = torch.squeeze(out_samples)
            zs_mean = torch.mean(z_samples, dim=0)

            # calculate entropy for the means of samples: - sum pc*log(pc)
            eps = 1e-10
            out_entropy.append(-torch.sum(out_mean * torch.log(out_mean + eps), dim=1).cpu().detach().numpy())

            # In contrast to the eval_dataset function, there is no split into correct or false values as the dataset
            # is unknown.
            for i in range(inputs.size(0)):
                idx = torch.argmax(out_mean[i]).item()
                out_mus[idx].append(out_mean[i][idx].item())
                out_sigmas[idx].append(out_std[i][idx].item())
                zs[idx].append(zs_mean[i].data)

    # stack latent activations into a tensor
    for i in range(len(zs)):
        if len(zs[i]) > 0:
            zs[i] = torch.stack(zs[i], dim=0)

    out_entropy = np.concatenate(out_entropy).ravel().tolist()

    # Return a dictionary of stored values.
    return {"out_mus": out_mus, "out_sigmas": out_sigmas, "zs": zs, "out_entropy": out_entropy}
