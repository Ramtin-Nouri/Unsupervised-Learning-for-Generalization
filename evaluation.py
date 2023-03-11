import torch
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt


def get_relative_confusion_matrix(confusion_matrix):
    """ Returns the confusion matrix with the values relative to the total number of predictions for each class. 
     confusion_matrix_relative = confusion_matrix / np.sum(confusion_matrix, axis=1)
    
    Args:
        confusion_matrix (np.array): The confusion matrix with the absolute values.
        
    Returns:
        np.array: The confusion matrix with the values relative to the total number of predictions for each class.
    """
    confusion_matrix_relative = np.zeros_like(confusion_matrix)
    for i in range(confusion_matrix.shape[0]):
        confusion_matrix_relative[i] = confusion_matrix[i] * 100 / np.sum(confusion_matrix[i]) if np.sum(
            confusion_matrix[i]) > 0 else 0

    return confusion_matrix_relative


def create_confusion_matrix_plt(plot_matrix, title, floating, dictionary):
    """ Creates a confusion matrix plot.

    Args:
        plot_matrix (np.array): The confusion matrix.
        title (str): The title of the plot.
        floating (bool): If True the values are floating point numbers, otherwise integers (percentage).
    """
    fig, ax = plt.subplots()
    im = ax.imshow(plot_matrix, vmin=0.0, vmax=np.max(plot_matrix))

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Frequency", rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(dictionary)), labels=dictionary)
    ax.set_yticks(np.arange(len(dictionary)), labels=dictionary)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(dictionary)):
        for j in range(len(dictionary)):
            my_formatter = "{0:2.2f}" if floating else "{0:4.0f}"
            text = ax.text(j, i, f"{my_formatter.format(plot_matrix[i, j])}{'%' if floating else ''}",
                           ha="center", va="center", color="w" if plot_matrix[i, j] < np.max(plot_matrix) / 2 else "0")

    ax.set_title(title)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    fig.tight_layout()
    fig.set_size_inches(24, 18)
    fig.tight_layout()
    return plt


def get_evaluation(model, data_loader, device, config, description=""):
    """ Evaluates the model on the given data loader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        data_loader (torch.utils.data.DataLoader): The data loader to evaluate on.
        device (torch.device): The device to use.
        config (dict): The configuration.
        description (str): The description of what is evaluated.

    Returns:
        float: The accuracy.
        np.array: The confusion matrix.
    """
    dictionary_size = config["dictionary_size"]
    sentence_length = config["sentence_length"]

    confusion_matrix = np.zeros((dictionary_size, dictionary_size))
    model.eval()

    with torch.no_grad():
        outputs = []
        labels = []
        correct_sentences = 0
        # Iterate over all batches and collect the outputs and labels.
        for batch in tqdm(data_loader, desc=description):
            if len(batch) == 2: #ARC-GEN
                frames_batch, label_batch = batch
                frames_batch = frames_batch.to(device=device)  # (N, L, c, w, h)
                mask = torch.ones(frames_batch.size(0), label_batch.size(1), device=frames_batch.device)
                multi_sentence_length = label_batch.size(1)
                output_batch = model(frames_batch, mask, multi_sentence_length) # (N, L, D)
            else:
                frames_batch, joints_batch, label_batch = batch
                frames_batch = frames_batch.to(device=device)  # (N, L, c, w, h)
                mask = torch.ones(frames_batch.size(0), label_batch.size(1), device=frames_batch.device)

                output_batch = model(frames_batch, mask)
                
            outputs.append(output_batch) # (data_size, N, L, D)
            labels.append(label_batch) # (data_size, N, L)


        if config["multi_sentence"]:
            for l in range(len(outputs)):
                # outputs[l] is (N, L, D) aka 1 sentence
                output = outputs[l]
                for n in range(output.shape[0]): # iterate over batch size

                    for i in range(outputs[l].shape[1] // 4):
                        # Get the output and label for the current batch.
                        _, action_outputs   = torch.max(output[n, 0, :], dim=0)
                        _, color_outputs    = torch.max(output[n, 1, :], dim=0)
                        _, material_outputs = torch.max(output[n, 2, :], dim=0)
                        _, object_outputs   = torch.max(output[n, 3, :], dim=0)
                        
                        action_labels   = int(labels[l][n][0].item())
                        color_labels    = int(labels[l][n][1].item())
                        material_labels = int(labels[l][n][2].item())
                        object_labels   = int(labels[l][n][3].item())

                        all_sentence_correct = False
                        confusion_matrix[action_labels, (action_outputs.item())] += 1
                        confusion_matrix[color_labels, (color_outputs.item())] += 1
                        confusion_matrix[material_labels, (material_outputs.item())] += 1
                        confusion_matrix[object_labels, (object_outputs.item())] += 1

                        action_correct = torch.sum(action_outputs == action_labels)
                        color_correct = torch.sum(color_outputs == color_labels)
                        material_correct = torch.sum(material_outputs == material_labels)
                        object_correct = torch.sum(object_outputs == object_labels)

                        all_sentence_correct &= action_correct and color_correct and object_correct and material_correct

                    if all_sentence_correct:
                        correct_sentences += 1
        else:
            outputs = torch.cat(outputs, dim=1) 
            labels = torch.cat(labels, dim=1)
            _, action_outputs = torch.max(outputs[:, 0, :], dim=1)
            _, color_outputs = torch.max(outputs[:, 1, :], dim=1)
            _, object_outputs = torch.max(outputs[:, 2, :], dim=1)

            for n in range(outputs.shape[0]):
                confusion_matrix[int(labels[n, 0].item()), (action_outputs[n].item())] += 1
                confusion_matrix[int(labels[n, 1].item()), (color_outputs[n].item())] += 1
                confusion_matrix[int(labels[n, 2].item()), (object_outputs[n].item())] += 1

                action_correct = torch.sum(action_outputs[n] == labels[n, 0])
                color_correct = torch.sum(color_outputs[n] == labels[n, 1])
                object_correct = torch.sum(object_outputs[n] == labels[n, 2])

                if action_correct and color_correct and object_correct:
                    correct_sentences += 1

    sentence_wise_accuracy = correct_sentences * 100 / len(data_loader.dataset)

    return confusion_matrix, sentence_wise_accuracy
