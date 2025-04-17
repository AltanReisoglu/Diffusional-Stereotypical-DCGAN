from generator import Generator,EMA
from discriminator import Discriminator2
from dataloader import train_dataloaded
from addNoise import Diffusion
from torch import nn
from torch import optim
import torch
from plot_images import plot_images
from tqdm import tqdm
from accelerate import Accelerator
import copy
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

device="cuda"
import torchvision.utils as vutils
import os
def save_generated_images(generator, diffusion, epoch, device, output_path, num_samples=16):
    generator.eval()
    with torch.no_grad():
        # Rastgele birkaç gerçek resimle t oluştur
        sample_images = next(iter(train_dataloaded))[0:num_samples].to(device)
        t = diffusion.sample_timesteps(sample_images.shape[0]).to(device)
        x_t, _ = diffusion.noise_images(sample_images, t)
        fake_images = generator(x_t, t)

        os.makedirs(output_path, exist_ok=True)
        vutils.save_image(
            fake_images,
            os.path.join(output_path, f"fake_epoch_{epoch+1}.png"),
            normalize=True,
            nrow=int(num_samples**0.5),
            value_range=(0, 1),
        )
    generator.train()
diffusion=Diffusion()
net_gen=Generator().to(device)
net_disc=Discriminator2().to(device)

#net_gen.apply(weights_init)
net_disc.apply(weights_init)

def train(num_epochs,discriminator_net,generator_net,optimizerD,optimizerG,train_loader,fake_label,real_label,criterion,output_path,num_test_samples,device):
    ema=EMA(0.995)
    ema_model = copy.deepcopy(generator_net).eval().requires_grad_(False)
    accelerator = Accelerator()
    discriminator_net, generator_net, optimizerD, optimizerG, train_loader = accelerator.prepare(
        discriminator_net, generator_net, optimizerD, optimizerG, train_loader
    )
    num_batches = len(train_loader)
    for epoch in range(num_epochs):
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)
        
        for i, (real_images) in enumerate(loop):
            
            
            batch_size_real_imgs = real_images.shape[0]

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            """ The standard process to train a DCGAN network is to first train
            the discriminator on the batch of samples.
            """
            discriminator_net.zero_grad()

            real_images = real_images.to(device)

            # First training on real image, hence fill it with 1
            # Create Labels
            

            """ The discriminator is used to classify real images (drawn from the training set)
            and fake images (produced by the generator).
            So, next, train the discriminator network on real images and real labels:
            """
            output = discriminator_net(real_images).view(-1, 1)
            real_label = torch.ones_like(output,device=device)
            fake_label = torch.zeros_like(output,device=device)
            loss_disc_real = criterion(output, real_label)
            
            accelerator.backward(loss_disc_real)

            D_x = output.mean().item()

            # Creating noise variables for the input to whole adversarial network
            t = diffusion.sample_timesteps(real_images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(real_images, t)

            # Generate a batch of fake images using the generator network
            fake_images = generator_net(x_t,t)

            # As now training on fake image, fill label with 0's
            

            # Now train Discriminator on fake images
            output = discriminator_net(fake_images.detach()).view(-1, 1)

            loss_disc_fake = criterion(output, fake_label)
            accelerator.backward(loss_disc_fake)

            D_G_z1 = output.mean().item()

            # Total Discriminator Loss
            loss_disc_total = loss_disc_real + loss_disc_fake

            optimizerD.step()


            generator_net.zero_grad()

            output = discriminator_net(fake_images)

            loss_generator = criterion(output, real_label)

            accelerator.backward(loss_generator)

            D_G_z2 = output.mean().item()

            optimizerG.step()
            ema.step_ema(ema_model, generator_net)
            loop.set_postfix({
            "Loss_D": loss_disc_total.item(),
            
            "D(x)": D_x,
            "D(G(z))": D_G_z2
            })
            if (i + 1) % 100 == 0:
                print(
                    "Epoch [{}/{}], step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, Discriminator - D(G(x)): {:.2f}, Generator - D(G(x)): {:.2f}".format(
                        epoch + 1,
                        num_epochs,
                        i + 1,
                        num_batches,
                        loss_disc_total.item(),
                        loss_generator.item(),
                        D_x,
                        D_G_z1,
                        D_G_z2,
                    )
                )
        if accelerator.is_main_process:
            generator_net.eval()
            save_generated_images(ema_model, diffusion, epoch, device, output_path, num_test_samples)
            print("resimler_kaydeidi")
            generator_net.train()


##########################################
# Initialize all the necessary variables
#########################################

batch_size = 32

output_path = "content"


# loss function
criterion = nn.BCELoss()

# optimizers
optimizerD = optim.Adam(net_disc.parameters(), lr=0.0001)
optimizerG = optim.Adam(net_gen.parameters(), lr=0.0001)

# initialize variables required for training
real_label = 1.0
fake_label = 0.0
# num_batches = len(train_loader)

num_test_samples = 16

num_epochs = 100

##########################################
# Execute the train Function
#########################################

train(
    num_epochs=num_epochs,
    discriminator_net=net_disc,
    generator_net=net_gen,
    optimizerD=optimizerD,
    optimizerG=optimizerG,
    train_loader=train_dataloaded,
    fake_label=fake_label,
    real_label=real_label,
    criterion=criterion,
    output_path=output_path,
    num_test_samples=num_test_samples,
    device=device,
    
)
