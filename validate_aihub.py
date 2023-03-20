from tqdm import tqdm
import numpy as np
import torch
from torch.nn.modules.distance import PairwiseDistance
from validate_on_LFW import evaluate_lfw
from plot import plot_roc_lfw, plot_accuracy_lfw


def validate_lfw(model, lfw_dataloader, model_architecture, epoch):
    model.eval()
    with torch.no_grad():
        l2_distance = PairwiseDistance(p=2)
        distances, labels = [], []

        print("Validating on LFW! ...")
        progress_bar = enumerate(tqdm(lfw_dataloader))

        for batch_index, (data_a, data_b, label) in progress_bar:
            data_a = data_a.cuda()
            data_b = data_b.cuda()

            output_a, output_b = model(data_a), model(data_b)
            distance = l2_distance.forward(output_a, output_b)  # Euclidean distance

            distances.append(distance.cpu().detach().numpy())
            labels.append(label.cpu().detach().numpy())

        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array([subdist for distance in distances for subdist in distance])

        true_positive_rate, false_positive_rate, precision, recall, accuracy, roc_auc, best_distances, \
        tar, far = evaluate_lfw(
            distances=distances,
            labels=labels,
            far_target=1e-3
        )
        # Print statistics and add to log
        print("Accuracy on LFW: {:.4f}+-{:.4f}\tPrecision {:.4f}+-{:.4f}\tRecall {:.4f}+-{:.4f}\t"
              "ROC Area Under Curve: {:.4f}\tBest distance threshold: {:.2f}+-{:.2f}\t"
              "TAR: {:.4f}+-{:.4f} @ FAR: {:.4f}".format(
                    np.mean(accuracy),
                    np.std(accuracy),
                    np.mean(precision),
                    np.std(precision),
                    np.mean(recall),
                    np.std(recall),
                    roc_auc,
                    np.mean(best_distances),
                    np.std(best_distances),
                    np.mean(tar),
                    np.std(tar),
                    np.mean(far)
                )
        )
        with open('logs/lfw_{}_log_triplet.txt'.format(model_architecture), 'a') as f:
            val_list = [
                epoch,
                np.mean(accuracy),
                np.std(accuracy),
                np.mean(precision),
                np.std(precision),
                np.mean(recall),
                np.std(recall),
                roc_auc,
                np.mean(best_distances),
                np.std(best_distances),
                np.mean(tar)
            ]
            log = '\t'.join(str(value) for value in val_list)
            f.writelines(log + '\n')

    try:
        # Plot ROC curve
        plot_roc_lfw(
            false_positive_rate=false_positive_rate,
            true_positive_rate=true_positive_rate,
            figure_name="plots/roc_plots/roc_{}_epoch_{}_triplet.png".format(model_architecture, epoch)
        )
        # Plot LFW accuracies plot
        plot_accuracy_lfw(
            log_file="logs/lfw_{}_log_triplet.txt".format(model_architecture),
            epochs=epoch,
            figure_name="plots/accuracies_plots/lfw_accuracies_{}_epoch_{}_triplet.png".format(model_architecture, epoch)
        )
    except Exception as e:
        print(e)

    return best_distances