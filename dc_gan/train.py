from generator import Generator
from discriminator import Discriminator2
from dataloader import train_dataloaded
from addNoise import Diffusion
from torch import nn
from torch import optim
import torch
from plot_images import plot_images
from tqdm import tqdm
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

device="cuda"
diffusion=Diffusion()
net_gen=Generator().to(device)
net_disc=Discriminator2().to(device)

#net_gen.apply(weights_init)
net_disc.apply(weights_init)

def train(num_epochs,discriminator_net,generator_net,optimizerD,optimizerG,train_loader,fake_label,real_label,criterion,output_path,num_test_samples,device):
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
            
            loss_disc_real.backward()

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
            loss_disc_fake.backward()

            D_G_z1 = output.mean().item()

            # Total Discriminator Loss
            loss_disc_total = loss_disc_real + loss_disc_fake

            optimizerD.step()

            ############################
            # (2) Update Generator network: maximize log(D(G(z)))
            ###########################

            """ When we train the generator network we have to
            freeze the discriminator network, as we have already trained it. """

            generator_net.zero_grad()

            # Now, set Image Label vector values equal to 1
            # To fool the Discriminator Network
            

            # After filling all labels with 1 (representing real labels), run discriminator network with fake images to fool it
            # To classify real images (drawn from the training set) and fakes images (produced by the generator).
            output = discriminator_net(fake_images)

            # And now after I tried to fool discriminator, check how much it was fooled.
            # so to the extent above output does not match with "labels" variable (which were all filed up with 1)
            # That will be the failure of Generator Network i.e. Generator Loss
            loss_generator = criterion(output, real_label)

            loss_generator.backward()

            D_G_z2 = output.mean().item()

            optimizerG.step()
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
    generator_net.eval()
    plot_images(
        epoch,
        output_path,
        num_test_samples,
        generator_net,
        device,
    )
    generator_net.train()


##########################################
# Initialize all the necessary variables
#########################################

batch_size = 32

output_path = "/content/"


# loss function
criterion = nn.BCELoss()

# optimizers
optimizerD = optim.Adam(net_disc.parameters(), lr=0.001,betas=(0.5,0.999))
optimizerG = optim.Adam(net_gen.parameters(), lr=0.001,betas=(0.5,0.999))

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