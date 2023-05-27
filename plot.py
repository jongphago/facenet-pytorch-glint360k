import matplotlib.pyplot as plt
from sklearn.metrics import auc


def plot_roc_lfw(false_positive_rate, true_positive_rate, figure_name="roc.png"):
    """Plots the Receiver Operating Characteristic (ROC) curve.

    Args:
        false_positive_rate: False positive rate
        true_positive_rate: True positive rate
        figure_name (str): Name of the image file of the resulting ROC curve plot.
    """
    roc_auc = auc(false_positive_rate, true_positive_rate)
    fig = plt.figure()
    plt.plot(
        false_positive_rate,
        true_positive_rate,
        color="red",
        lw=2,
        label="ROC Curve (area = {:.4f})".format(roc_auc),
    )
    plt.plot([0, 1], [0, 1], color="blue", lw=2, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    fig.savefig(figure_name, dpi=fig.dpi)
    plt.close()


def plot_accuracy_lfw(log_file, epochs, dataset="LFW", figure_name="lfw_accuracy.png"):
    """Plots the accuracies on the Labeled Faces in the Wild dataset over the training epochs.

    Args:
        log_file (str): Path of the log file containing the lfw accuracy values to be plotted.
        epochs (int): Number of training epochs finished.
        figure_name (str): Name of the image file of the resulting lfw accuracies plot.
    """
    with open(log_file, "r") as f:
        lines = f.readlines()
        epoch_list = [int(line.split("\t")[0]) for line in lines]
        accuracy_list = [round(float(line.split("\t")[1]), 2) for line in lines]

        fig = plt.figure()
        plt.plot(epoch_list, accuracy_list, color="red", label=f"{dataset} Accuracy")
        plt.ylim([0.0, 1.05])
        plt.xlim([0, epochs + 1])
        plt.xlabel("Epoch")
        plt.ylabel(f"{dataset} Accuracy")
        plt.title(f"{dataset} Accuracies plot")
        plt.legend(loc="lower right")
        fig.savefig(figure_name, dpi=fig.dpi)
        plt.close()


def plot_accuracy_aihub(
    log_file, epochs, dataset="AIHub", figure_name="aihub_accuracy.png"
):
    """
    AIHub 데이터셋의 정확도를 시각화하여 그래프로 저장합니다.

    Parameters:
    log_file (str): 로그 파일의 경로
    epochs (int): 총 에포크 수
    dataset (str, optional): 데이터셋의 이름 (기본값: "AIHub")
    figure_name (str, optional): 저장할 그래프 이미지 파일의 이름 (기본값: "aihub_accuracy.png")

    Returns:
    None
    """
    with open(log_file, "r") as f:
        lines = f.readlines()
        # epoch_list = [int(line.split("\t")[0]) for line in lines]
        accuracy_list = [round(float(line.split("\t")[1]), 2) for line in lines]
        fig = plt.figure()
        plt.plot(accuracy_list, color="red", label=f"{dataset} Accuracy")
        plt.ylim([0.0, 1.05])
        # plt.xlim([0, epochs + 1])
        plt.xlabel("Epoch")
        plt.ylabel(f"{dataset} Accuracy")
        plt.title(f"{dataset} Accuracies plot")
        plt.legend(loc="lower right")
        fig.savefig(figure_name, dpi=fig.dpi)
        plt.close()
