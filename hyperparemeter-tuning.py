import time
from options.train_options import TrainOptions
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer

def train_one_epoch(opt, structural_weight):
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print(f'The number of training images = {dataset_size}')

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.structural_weight = structural_weight  # Set the structural weight
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    
    epoch_start_time = time.time()  # timer for entire epoch
    iter_data_time = time.time()    # timer for data loading per iteration
    epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
    visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
    model.update_learning_rate()    # update learning rates in the beginning of every epoch.
    
    total_loss = 0  # To keep track of the total loss for the epoch
    
    for i, data in enumerate(dataset):  # inner loop within one epoch
        iter_start_time = time.time()  # timer for computation per iteration
        if i % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time

        epoch_iter += opt.batch_size
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

        if i % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
            save_result = i % opt.update_html_freq == 0
            model.compute_visuals()
            visualizer.display_current_results(model.get_current_visuals(), 1, save_result)

        if i % opt.print_freq == 0:    # print training losses and save logging information to the disk
            losses = model.get_current_losses()
            t_comp = (time.time() - iter_start_time) / opt.batch_size
            visualizer.print_current_losses(1, epoch_iter, losses, t_comp, t_data)
            if opt.display_id > 0:
                visualizer.plot_current_losses(1, float(epoch_iter) / dataset_size, losses)
        
        # Accumulate total loss
        total_loss += sum(model.get_current_losses().values())

        iter_data_time = time.time()
    
    print(f'End of epoch 1 \t Time Taken: {time.time() - epoch_start_time} sec')
    return total_loss / dataset_size  # Return average loss per iteration

def hyperparameter_tuning():
    opt = TrainOptions().parse()   # get training options
    opt.n_epochs = 1  # Set to train for only one epoch
    opt.n_epochs_decay = 0  # No decay for just one epoch
    
    # Define the hyperparameter grid
    structural_weights = [0.1, 0.2, 0.3, 0.4, 0.5]
    
    best_loss = float('inf')
    best_weight = None
    
    for weight in structural_weights:
        print(f"Training with structural weight: {weight}")
        avg_loss = train_one_epoch(opt, weight)
        
        print(f"Average loss for weight {weight}: {avg_loss}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_weight = weight
    
    print(f"Best structural weight: {best_weight}")
    print(f"Best average loss: {best_loss}")

if __name__ == '__main__':
    hyperparameter_tuning()