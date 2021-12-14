import os
from typing import List, Union
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import torch.backends.cudnn as cudnn

from models import *


def get_dataloader():
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    return DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)


def restore_weights(
    net: Union[VGG, CustomVGG], checkpoint_dir: str, p_val: int, device
):
    # Restore model weights
    net = net.to(device)
    if device == "cuda":
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    checkpoint_file = f"{checkpoint_dir}/ckpt.pth"
    checkpoint = torch.load(checkpoint_file)
    net.load_state_dict(checkpoint["net"])
    print("Model restored from", checkpoint_file)
    print(net)
    return net


def export_tensors(outputs: List[torch.Tensor], directory: str, suffix="plain"):
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i, out in enumerate(outputs):
        print(out.shape)
        np.save(f"{directory}/{i}.{suffix}", out.cpu().numpy())


def save_quantized_model_weights(model, directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)

    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        weights = model.state_dict()[param_tensor]
        if torch.is_tensor(weights):
            print(param_tensor, "\t", weights.size())
            np.save(f"{directory}/{param_tensor}", weights.cpu().numpy())


def save_infer(
    net: Union[VGG, CustomVGG], checkpoint_dir: str, device, limit=-1, quant_dtype=None,
):
    prefix = "checkpoint-"
    start = checkpoint_dir.find(prefix)
    if start == 0:
        export_dir = checkpoint_dir[len(prefix) :]
    export_dir = f"export/{export_dir}"
    print("export results to", export_dir)
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    exp_inputs = []
    conv_out = []
    norm_out = []
    relu_out = []
    outputs = []

    def conv_hook_fn(m, i, o: torch.Tensor):
        conv_out.append(o.clone().detach())

    def norm_hook_fn(m, i, o: torch.Tensor):
        norm_out.append(o.clone().detach())

    def relu_hook_fn(m, i, o: torch.Tensor):
        relu_out.append(o.clone().detach())

    net.features[0].register_forward_hook(conv_hook_fn)
    net.features[1].register_forward_hook(norm_hook_fn)
    net.features[2].register_forward_hook(relu_hook_fn)

    net = restore_weights(
        net=net, checkpoint_dir=checkpoint_dir, p_val=p_val, device=device
    )
    suffix = "plain"
    if quant_dtype is not None:
        device = "cpu"
        net = net.module.to(device)
        torch.quantization.quantize_dynamic(net, dtype=quant_dtype, inplace=True)
        save_quantized_model_weights(net, directory=f"{export_dir}/weights")
        suffix = str(quant_dtype)
    net.eval()
    testloader = get_dataloader()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if limit >= 0 and batch_idx >= limit:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            exp_inputs.append(inputs.clone().detach())
            output = net(inputs)
            outputs.append(output.clone().detach())

    export_tensors(exp_inputs, f"{export_dir}/input", suffix=suffix)
    export_tensors(conv_out, f"{export_dir}/conv", suffix=suffix)
    export_tensors(norm_out, f"{export_dir}/norm", suffix=suffix)
    export_tensors(relu_out, f"{export_dir}/relu", suffix=suffix)
    export_tensors(outputs, f"{export_dir}/final", suffix=suffix)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Use device", device)
    p_val = -1
    checkpoint_dir = f"checkpoint-cvgg16a-p{p_val}"
    print("Custom VGG16", "p=", p_val)
    net = CustomVGG("VGG16A", p_value=p_val)
    save_infer(
        net=net, checkpoint_dir=checkpoint_dir, device=device, limit=10,
    )
    save_infer(
        net=net,
        checkpoint_dir=checkpoint_dir,
        device=device,
        limit=10,
        quant_dtype=torch.qint8,
    )
