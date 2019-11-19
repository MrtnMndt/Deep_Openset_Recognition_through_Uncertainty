########################
# Importing libraries
########################
# System libraries
import os
import random
from time import gmtime, strftime

# Tensorboard for PyTorch logging and visualization
from torch.utils.tensorboard import SummaryWriter

# Torch libraries
import torch
import torch.backends.cudnn as cudnn

# Custom library
import lib.Models.architectures as architectures
import lib.Datasets.datasets as datasets
from lib.Models.initialization import WeightInit
from lib.cmdparser import parser
from lib.Utility.utils import save_checkpoint
from lib.Utility.visualization import args_to_tensorboard


# Comment this if CUDNN benchmarking is not desired
cudnn.benchmark = True


def main():
    # Command line options
    args = parser.parse_args()
    print("Command line options:")
    for arg in vars(args):
        print(arg, getattr(args, arg))

    # import the correct loss and training functions depending which model to optimize
    # TODO: these could easily be refactored into one function, but we kept it this way for modularity
    if args.train_var:
        if args.joint:
            from lib.Training.train import train_var_joint as train
            from lib.Training.validate import validate_var_joint as validate
            from lib.Training.loss_functions import var_loss_function_joint as criterion
        else:
            from lib.Training.train import train_var as train
            from lib.Training.validate import validate_var as validate
            from lib.Training.loss_functions import var_loss_function as criterion
    else:
        if args.joint:
            from lib.Training.train import train_joint as train
            from lib.Training.validate import validate_joint as validate
            from lib.Training.loss_functions import loss_function_joint as criterion
        else:
            from lib.Training.train import train as train
            from lib.Training.validate import validate as validate
            from lib.Training.loss_functions import loss_function as criterion

    # Check whether GPU is available and can be used
    # if CUDA is found then device is set accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Launch a writer for the tensorboard summary writer instance
    save_path = 'runs/' + strftime("%Y-%m-%d_%H-%M-%S", gmtime()) + '_' + args.dataset + '_' + args.architecture +\
                '_dropout_' + str(args.dropout)

    if args.train_var:
        save_path += '_variational_samples_' + str(args.var_samples) + '_latent_dim_' + str(args.var_latent_dim)

    if args.joint:
        save_path += '_joint'

    # if we are resuming a previous training, note it in the name
    if args.resume:
        save_path = save_path + '_resumed'
    writer = SummaryWriter(save_path)

    # saving the parsed args to file
    log_file = os.path.join(save_path, "stdout")
    log = open(log_file, "a")
    for arg in vars(args):
        log.write(arg + ':' + str(getattr(args, arg)) + '\n')

    # Dataset loading
    data_init_method = getattr(datasets, args.dataset)
    dataset = data_init_method(torch.cuda.is_available(), args)
    # get the number of classes from the class dictionary
    num_classes = dataset.num_classes

    # add command line options to TensorBoard
    args_to_tensorboard(writer, args)

    log.close()

    # Get a sample input from the data loader to infer color channels/size
    net_input, _ = next(iter(dataset.train_loader))
    # get the amount of color channels in the input images
    num_colors = net_input.size(1)

    # import model from architectures class
    net_init_method = getattr(architectures, args.architecture)

    # build the model
    model = net_init_method(device, num_classes, num_colors, args)

    # Parallel container for multi GPU use and cast to available device
    model = torch.nn.DataParallel(model).to(device)
    print(model)

    # Initialize the weights of the model, by default according to He et al.
    print("Initializing network with: " + args.weight_init)
    WeightInitializer = WeightInit(args.weight_init)
    WeightInitializer.init_model(model)

    # Define optimizer and loss function (criterion)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    epoch = 0
    best_prec = 0
    best_loss = random.getrandbits(128)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            epoch = checkpoint['epoch']
            best_prec = checkpoint['best_prec']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # optimize until final amount of epochs is reached.
    while epoch < args.epochs:
        # train
        train(dataset, model, criterion, epoch, optimizer, writer, device, args)

        # evaluate on validation set
        prec, loss = validate(dataset, model, criterion, epoch, writer, device, args)

        # remember best prec@1 and save checkpoint
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        best_prec = max(prec, best_prec)
        save_checkpoint({'epoch': epoch,
                         'arch': args.architecture,
                         'state_dict': model.state_dict(),
                         'best_prec': best_prec,
                         'best_loss': best_loss,
                         'optimizer': optimizer.state_dict()},
                        is_best, save_path)

        # increment epoch counters
        epoch += 1

    writer.close()


if __name__ == '__main__':     
    main()
