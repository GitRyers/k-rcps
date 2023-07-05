import os

import torch
from absl import app, flags
from dataset import get_dataset
from ml_collections.config_flags import config_flags
from models import utils as mutils
from torch.utils.data import DataLoader
from tqdm import tqdm

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", None, "Configuration", lock_config=False)
flags.DEFINE_string("gpu", "0", "GPU to use")
flags.DEFINE_string("workdir", "./", "Working directory")


def main(_):
    config = FLAGS.config
    gpu = FLAGS.gpu
    workdir = FLAGS.workdir

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    denoising_dir = os.path.join(workdir, "denoising")
    denoising_dataset_dir = os.path.join(denoising_dir, config.data.name)
    perturbed_dir = os.path.join(denoising_dataset_dir, "perturbed")
    denoised_dir = os.path.join(denoising_dataset_dir, config.name)
    os.makedirs(denoised_dir, exist_ok=True)

    batch_size = 1
    config.data.return_img_id = True
    _, dataset = get_dataset(config)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    if config.data.dataset == "CelebA":
        sigma0 = 1.0
    if config.data.dataset == "AbdomenCT1K":
        sigma0 = 0.4

    def enable_dropout(model):
        for m in model.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()

    from models import im2im_ncsnpp

    model = mutils.get_model(config.model.im2im_config, checkpoint=True)
    model = model.to(device)
    model.eval()
    enable_dropout(model)
    torch.set_grad_enabled(False)

    sampling_batch_size = 32
    total_samples = 128
    for _, data in enumerate(tqdm(dataloader)):
        _, img_id = data

        y = torch.stack(
            [torch.load(os.path.join(perturbed_dir, f"{_id}.pt")) for _id in img_id],
            dim=0,
        )
        y = y.to(device)
        y = y.repeat(sampling_batch_size, 1, 1, 1)

        sigma_t = torch.tensor(y.size(0) * [sigma0], device=device)

        samples = []
        for _ in range(total_samples // sampling_batch_size):
            output = model(y, sigma_t)
            output = output[:, 0]

            sampled = output.view(
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


if __name__ == "__main__":
    app.run(main)
