import argparse
from datasets import create_dataset
from utils import parse_configuration
from models import create_model
import os
from utils.visualizer import Visualizer
import matplotlib.pyplot as plt
import numpy as np
import torch

"""Performs validation of a specified model.

Input params:
    config_file: Either a string with the path to the JSON
        system-specific config file or a dictionary containing
        the system-specific, dataset-specific and
        model-specific settings.
"""
def validate(configuration):

    print('Initializing dataset...')
    val_dataset = create_dataset(configuration['val_dataset_params'])
    val_dataset_size = len(val_dataset)
    print('The number of validation samples = {0}'.format(val_dataset_size))

    print('Initializing model...')
    model = create_model(configuration['model_params'])
    model.setup()
    autoencoder_weights = torch.load(configuration['model_params']['autoencoder_path'])
    model.load_autoencoder(autoencoder_weights)
    model.eval()

    print('Initializing visualization...')
    visualizer = Visualizer(configuration['visualization_params_validation'])   # create a visualizer that displays images and plots

    model.pre_epoch_callback(configuration['model_params']['load_checkpoint'])

    patch_size = configuration['model_params']['patch_size']
    input_size = configuration['train_dataset_params']['input_size']

    #Loops through all validation data and runs though model
    for i, data in enumerate(val_dataset):
        output = torch.empty((3, input_size[0], input_size[1]))
        noise = torch.empty((3, input_size[0], input_size[1])).cuda()
        for ip in range(int(float(input_size[0])/patch_size)):
            for jp in range(int(float(input_size[1])/patch_size)):
                cur_data = [data[0][:, :, ip*patch_size:(ip+1)*patch_size, jp*patch_size:(jp+1)*patch_size],
                            data[1][:, :, ip*patch_size:(ip+1)*patch_size, jp*patch_size:(jp+1)*patch_size]]
                model.set_input(cur_data)         # unpack data from dataset and apply preprocessing
                model.test()

                cur_noise = model.output[0] - model.input[0]
                output[:, ip*patch_size:(ip+1)*patch_size, jp*patch_size:(jp+1)*patch_size] = model.output[0]

                noise[:, ip*patch_size:(ip+1)*patch_size, jp*patch_size:(jp+1)*patch_size] = cur_noise

        img = data[0][0].permute(1, 2, 0).cpu().detach().numpy()
        out_img = output.permute(1, 2, 0).cpu().detach().numpy()
        noise = noise.permute(1, 2, 0).cpu().detach().numpy()
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(img)
        axs[0].set_title('Input')
        axs[1].imshow(out_img)
        axs[1].set_title('Output')
        axs[2].imshow(noise)
        axs[2].set_title('Noise')
        plt.savefig("./plots/encoder_layer_only/best_denoised/epoch_{}_img_{}.png".format(configuration['model_params']['scheduler_epoch'], i))
        plt.close()

    #Where results are calculated and visualized
    # model.post_epoch_callback(configuration['model_params']['load_checkpoint'], visualizer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform model validation.')
    parser.add_argument('configfile', help='path to the configfile')

    args = parser.parse_args()

    print('Reading config file...')
    configuration = parse_configuration(args.configfile)
    if (configuration['model_params']['load_checkpoint'] == -2):
        for epoch in range(configuration['model_params']['epoch_list'][0], configuration['model_params']['epoch_list'][1]):
            configuration['model_params']['load_checkpoint'] = epoch
            validate(configuration)
    else:
        validate(configuration)
