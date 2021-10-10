from typing import Callable, Union
from matplotlib.axes import Axes
import torch
import torch.nn as nn
import numpy as np
from models.custom import DentReLUFunction
import matplotlib.pyplot as plt
import utils
import models

plt.style.use("classic")


def _ax_plot(ax_func: Callable[[Axes], None], name):
    plt.figure(figsize=(5, 5))
    fig, ax = plt.subplots()
    # fig.set_size_inches(5, 5)
    ax_func(ax)
    # ax.grid(True)
    fig.savefig(f"plots/{name}.png", bbox_inches="tight")
    fig.savefig(f"plots/{name}.pdf", bbox_inches="tight")
    plt.close()


def plot_formula():
    def ax_func(ax):
        # Move the left and bottom spines to x = 0 and y = 0, respectively.
        ax.spines[["left", "bottom"]].set_position(("data", 0))
        # Hide the top and right spines.
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_ticks_position("bottom")
        ax.yaxis.set_ticks_position("left")
        x = torch.tensor(np.arange(-1, 1, 0.0005))
        y = DentReLUFunction.apply(x, -0.666)
        ax.set_aspect("equal")
        ax.plot(x, y)

    _ax_plot(ax_func, "formula")


def plot_dist():
    def ax_func(ax):
        x = torch.tensor(np.arange(-1, 1, 0.01))
        y1 = torch.relu(x)
        p_values = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]
        dist = []
        for p in p_values:
            p = (2 - p) / p
            y2 = DentReLUFunction.apply(x, p)
            d = torch.dist(y1, y2, p=2)
            dist.append(d)
        ax.stem(p_values, dist, markerfmt="x")
        ax.set_xticks(p_values)
        ax.set_ylim(1, 6)
        # ax.set_xticklabels(ax.get_xticks(), rotation=90)

    _ax_plot(ax_func, "dist")


def plot_train_test():
    stock_hist_acc = np.loadtxt("stock_hist_acc.txt", delimiter=",")
    stock_hist_loss = np.loadtxt("stock_hist_loss.txt", delimiter=",")
    p1_hist_acc = np.loadtxt("p1_hist_acc.txt", delimiter=",")
    p1_hist_loss = np.loadtxt("p1_hist_loss.txt", delimiter=",")
    p2_hist_acc = np.loadtxt("p2_hist_acc.txt", delimiter=",")
    p2_hist_loss = np.loadtxt("p2_hist_loss.txt", delimiter=",")

    def ax_func(ax):
        epochs = np.arange(1, len(stock_hist_acc) + 1)
        ax.plot(epochs, stock_hist_acc / 100, label="Relu_acc")
        ax.plot(epochs, stock_hist_loss, label="Relu_loss")
        ax.plot(epochs, p1_hist_acc / 100, label="p=3_acc")
        ax.plot(epochs, p1_hist_loss, label="p=3_loss")
        ax.plot(epochs, p2_hist_acc / 100, label="p=5_acc")
        ax.plot(epochs, p2_hist_loss, label="p=5_loss")
        ax.set_xlim(0, len(stock_hist_acc) + 1)
        ax.legend()

    _ax_plot(ax_func, "train_test")


def plot_hist(net: Union[models.VGG, models.CustomVGG], checkpoint, name):
    conv_out = []
    norm_out = []
    relu_out = []

    def conv_hook_fn(m, i, o: torch.Tensor):
        print(m)
        conv_out.append(o.clone().detach())

    def norm_hook_fn(m, i, o: torch.Tensor):
        print(m)
        norm_out.append(o.clone().detach())

    def relu_hook_fn(m, i, o: torch.Tensor):
        print(m)
        relu_out.append(o.clone().detach())

    testloader = utils.prepare_test_data()
    net.features[0].register_forward_hook(conv_hook_fn)
    net.features[1].register_forward_hook(norm_hook_fn)
    net.features[2].register_forward_hook(relu_hook_fn)
    utils.test_model_vis(net, checkpoint, testloader)
    conv_out = torch.cat(conv_out)
    print(conv_out.shape)
    norm_out = torch.cat(norm_out)
    print(norm_out.shape)
    relu_out = torch.cat(relu_out)
    print(relu_out.shape)

    def conv_ax_func(ax):
        ax.set_xlim(-6, 6)
        ax.set_yscale("log")
        h0 = conv_out.cpu().numpy().ravel()
        ax.hist(h0, label=["conv"], bins=10)
        ax.legend()
        ax.set_title(f"Conv output: min={np.min(h0):.4f}, max={np.max(h0):.4f}")
        np.savetxt(f"{name}-conv.txt", h0, fmt="%.8f")

    def norm_ax_func(ax):
        ax.set_xlim(-6, 6)
        ax.set_yscale("log")
        h1 = norm_out.cpu().numpy().ravel()
        ax.hist(h1, label=["norm"], bins=10)
        ax.legend()
        ax.set_title(f"Norm output: min={np.min(h1):.4f}, max={np.max(h1):.4f}")
        np.savetxt(f"{name}-norm.txt", h1, fmt="%.8f")

    def relu_ax_func(ax):
        p_val = 3 if isinstance(net, models.CustomVGG) and net.p_value == -1 else 5
        label = "Relu" if isinstance(net, models.VGG) else f"DRelu p={p_val}"
        ax.set_xlim(-6, 6)
        ax.set_yscale("log")
        h2 = relu_out.cpu().numpy().ravel()
        ax.hist([h2], label=[label.lower()], bins=10)
        ax.legend()
        ax.set_title(f"{label} output: min={np.min(h2):.4f}, max={np.max(h2):.4f}")
        np.savetxt(f"{name}-relu.txt", h2, fmt="%.8f")

    _ax_plot(conv_ax_func, f"{name}-conv")
    _ax_plot(norm_ax_func, f"{name}-norm")
    _ax_plot(relu_ax_func, f"{name}-relu")


# plot_formula()
# plot_dist()
# plot_train_test()
plot_hist(models.VGG("VGG16"), "checkpoint-vgg16-stock", "hist")
plot_hist(models.CustomVGG("VGG16", p_value=-1), "checkpoint-cvgg16-p-1", "hist-p-1")
plot_hist(
    models.CustomVGG("VGG16", p_value=-1.8), "checkpoint-cvgg16-p-1.8", "hist-p-1.8"
)
