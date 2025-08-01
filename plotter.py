from matplotlib import pyplot as plt
import seaborn as sn

def show_images_from_dataset(dataset:list,nrows:int,ncols:int) -> None:
    plt.figure(figsize=(5*nrows, 5*ncols))
    for i in range(nrows*ncols):
        plt.subplot(nrows, ncols, i + 1)
        plt.axis(True)
        plt.imshow(dataset[i], cmap="gray")
        plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.show()


def _plot_results(metrics, title=None, ylabel=None, ylim=None, metric_name=None, color=None):
    fig, ax = plt.subplots(figsize=(15, 4))

    if not (isinstance(metric_name, list) or isinstance(metric_name, tuple)):
        metrics = [metrics, ]
        metric_name = [metric_name, ]

    for idx, metric in enumerate(metrics):
        ax.plot(metric, color=color[idx])

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xlim([0, 20])
    plt.ylim(ylim)
    # Tailor x-axis tick marks
    # ax.xaxis.set_major_locator(MultipleLocator(5))
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    # ax.xaxis.set_minor_locator(MultipleLocator(1))
    plt.grid(True)
    plt.legend(metric_name)
    plt.show()
    plt.close()
    plt.show()

def show_results(training_results):
    # Retrieve training results.
    train_loss = training_results.history["loss"]
    train_acc = training_results.history["accuracy"]
    valid_loss = training_results.history["val_loss"]
    valid_acc = training_results.history["val_accuracy"]

    _plot_results(
        [train_loss, valid_loss],
        ylabel="Loss",
        ylim=[0.0, 0.5],
        metric_name=["Training Loss", "Validation Loss"],
        color=["g", "b"],
    )

    _plot_results(
        [train_acc, valid_acc],
        ylabel="Accuracy",
        ylim=[0.9, 1.0],
        metric_name=["Training Accuracy", "Validation Accuracy"],
        color=["g", "b"],
    )