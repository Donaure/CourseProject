import os
import torch
import torch.nn.functional as F
from vae import VAE
from encoder import Encoder
from decoder import Decoder
from torch import optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import argparse
from load_data import load_data
from torchvision.utils import save_image


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expname", type=str, default='3layer_h2048_h512_lat64')
    parser.add_argument("--input_dim", type=int, default=28 * 28)
    parser.add_argument("--h1_dim", type=int, default=2048, help="hidden layer1 dimension")
    parser.add_argument("--h2_dim", type=int, default=512, help="hidden layer2 dimension")
    parser.add_argument("--latent_dim", type=int, default=64, help="latent dimension")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--i_print", type=int, default=100)
    parser.add_argument("--beta", type=float, default=1.0, help="regularization coefficient of kl loss")
    return parser.parse_args()


def main():
    args = config_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter(os.path.join('exp', args.expname))

    for arg in vars(args):
        writer.add_text('args', f"{arg}: {getattr(args, arg)}")

    encoder = Encoder(input_dim=args.input_dim,
                      h1_dim=args.h1_dim,
                      h2_dim=args.h2_dim,
                      latent_dim=args.latent_dim).to(device)

    decoder = Decoder(latent_dim=args.latent_dim,
                      h1_dim=args.h1_dim,
                      h2_dim=args.h2_dim,
                      input_dim=args.input_dim).to(device)

    model = VAE(encoder, decoder).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_loader, test_loader = load_data(args.batch_size)

    total_loss, total_rec_loss, total_kl_loss = [], [], []

    print("Start Training ............")
    for i in tqdm(range(args.epoch)):
        model.train()
        # args.beta = beta * min(4*(1+i) / args.epoch, 1.0)  # apply dynamic beta
        train_loss, rec_loss, kl_loss = 0, 0, 0
        for idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            output, mean, logvar = model(data)
            optimizer.zero_grad()

            # compute reconstruction loss and kl divergence loss
            recon_loss = F.binary_cross_entropy(output, data.view(-1, args.input_dim), reduction='sum')
            rec_loss += recon_loss.item()
            kl_div_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            kl_loss += kl_div_loss.item()

            loss = recon_loss + args.beta * kl_div_loss
            train_loss += loss.item()

            loss.backward()
            optimizer.step()
            if (idx + 1) % args.i_print == 0:
                print(f"  Iter[{idx + 1}] loss:{loss.item():.5f}, rec loss:{recon_loss.item():.5f}, kl loss:{kl_div_loss.item():.5f}")

        total_loss.append(train_loss / len(train_loader))
        total_rec_loss.append(rec_loss / len(train_loader))
        total_kl_loss.append(kl_loss / len(train_loader))

        print(f"  Epoch[{i + 1}]: loss: {total_loss[-1]:.5f}, rec loss: {total_rec_loss[-1]:.5f}, kl loss: {total_kl_loss[-1]:.5f}")
        writer.add_scalar('train loss', total_loss[-1], i)
        writer.add_scalar('train rec loss', total_rec_loss[-1], i)
        writer.add_scalar('train kl loss', total_kl_loss[-1], i)

    # test
    print("Start Testing ............")
    model.eval()
    test_loss = 0
    for idx, (data, label) in enumerate(test_loader):
        data = data.to(device)
        # visualization first batch
        if idx == 0:
            output, mean, logvar = model(data)
            nelem = data.size(0)
            nrow = 10
            save_image(data.view(nelem, 1, 28, 28), './images/image_0_gt' + '.png', nrow=nrow)
            save_image(output.view(nelem, 1, 28, 28), './images/image_0_gen' + '.png', nrow=nrow)
            output.view(-1, args.input_dim)
        else:
            output, mean, logvar = model(data)
        recon_loss = F.binary_cross_entropy(output, data.view(-1, args.input_dim), reduction='sum')
        test_loss += recon_loss.item()
    writer.add_scalar('test loss', test_loss / len(test_loader))

    # visualization
    if args.latent_dim == 1:
        z = torch.linspace(-5, 5, 200).to(device).unsqueeze(-1)
        out = model.decode(z).view(200, 1, 28, 28)
        save_image(out, './images/vis_1dim' + '.png', nrow=20)

    elif args.latent_dim == 2:
        z = torch.linspace(-5, 5, 20).to(device)
        x, y = torch.meshgrid(z, z)
        out = model.decode(torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1).reshape(-1, 2))
        out = out.view(400, 1, 28, 28)
        save_image(out, './images/vis_2dim' + '.png', nrow=20)

    print("Done!")
    writer.close()


if __name__ == '__main__':
    main()
