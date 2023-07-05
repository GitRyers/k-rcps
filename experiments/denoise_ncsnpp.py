import os

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from absl import app, flags
from dataset import get_dataset
from ml_collections.config_flags import config_flags
from models import ncsnpp
from models import utils as mutils
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Configuration", lock_config=True)
flags.DEFINE_string("workdir", "./", "Working directory")
flags.DEFINE_string("gpu", "0,1,2,3,4,5,6,7", "GPU(s) to use")


def common(device, world_size, config, workdir, batch_size=1):
    # setup process
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group("nccl", rank=device, world_size=world_size)
    torch.cuda.set_device(device)

    # setup dirs
    denoising_dir = os.path.join(workdir, "denoising")
    denoising_dataset_dir = os.path.join(denoising_dir, config.data.name)
    perturbed_dir = os.path.join(denoising_dataset_dir, "perturbed")
    denoised_dir = os.path.join(os.path.join(denoising_dataset_dir, config.name))
    os.makedirs(denoised_dir, exist_ok=True)

    # setup dataset
    config.data.return_img_id = True
    _, dataset = get_dataset(config)
    sampler = DistributedSampler(dataset, shuffle=False)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, num_workers=4
    )

    if config.data.dataset == "CelebA":
        sigma0 = 1.0
    if config.data.dataset == "AbdomenCT-1K":
        sigma0 = 0.4

    # setup model
    model = mutils.get_model(config, checkpoint=True)
    model = model.to(device)
    model.eval()
    model = DDP(model, device_ids=[device], output_device=device)
    torch.set_grad_enabled(False)

    return dataloader, sigma0, model, perturbed_dir, denoised_dir


def sample(rank, world_size, config, workdir, batch_size=1, N=300):
    device = rank
    dataloader, sigma0, model, perturbed_dir, denoised_dir = common(
        device, world_size, config, workdir, batch_size=batch_size
    )

    # Initialize SDE
    sigma_min, sigma_max = config.model.sigma_min, config.model.sigma_max
    sigma = lambda t: sigma_min * (sigma_max / sigma_min) ** t
    diffusion = lambda t: sigma(t) * torch.sqrt(
        torch.tensor(2 * (np.log(sigma_max) - np.log(sigma_min)), device=device)
    )

    config.model.num_scales = N
    config.model.T = 1.0

    dt = config.model.T / config.model.num_scales
    t = torch.linspace(
        config.model.T, config.model.eps, config.model.num_scales, device=device
    )
    t0 = (np.log(sigma0) - np.log(sigma_min)) / (np.log(sigma_max) - np.log(sigma_min))

    score_fn = lambda x, t: model(x, sigma(t))
    _t = t[t < t0]

    sampling_batch_size = config.model.sampling_batch_size
    total_samples = config.model.total_samples
    for _, data in enumerate(tqdm(dataloader)):
        _, img_id = data

        y = torch.stack(
            [torch.load(os.path.join(perturbed_dir, f"{_id}.pt")) for _id in img_id],
            dim=0,
        )
        y = y.to(device)

        y = y.repeat(sampling_batch_size, 1, 1, 1)

        samples = []
        for _ in range(total_samples // sampling_batch_size):
            x = y.clone()
            for _, tt in enumerate(_t):
                noise = torch.randn_like(x)

                score = score_fn(x, tt.repeat(x.size(0)))

                sigma_tt = sigma(tt)
                diffusion_tt = diffusion(tt)
                x = (
                    x
                    + diffusion_tt**2
                    * (score + (y - x) / (sigma0**2 - sigma_tt**2))
                    * dt
                    + diffusion_tt * np.sqrt(dt) * noise
                )

            sampled = x.view(
                batch_size,
                sampling_batch_size,
                config.data.num_channels,
                config.data.image_size,
                config.data.image_size,
            )
            samples.append(sampled.cpu())

        sampled = torch.cat(samples, dim=1)
        for _id, _sampled in zip(img_id, sampled):
            torch.save(_sampled, os.path.join(denoised_dir, f"{_id}.pt"))


def main(_):
    config = FLAGS.config
    workdir = FLAGS.workdir
    gpu = FLAGS.gpu

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    world_size = len(gpu.split(","))

    mp.spawn(sample, args=(world_size, config, workdir), nprocs=world_size)


if __name__ == "__main__":
    app.run(main)
