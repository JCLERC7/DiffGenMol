import torch
import utils
from diffusion.diffusion_1d import Diffusion1D
from model.unet_1d import Unet1D
from rdkit import Chem
from rdkit.Chem import Draw
from torch.optim import Adam

class Trainer1D():
    """
    Trainer 1D class
    """
    def __init__(self, data_loader, timesteps, epochs, lr, unet_dim, unet_channels, unet_cond_drop_prob, results_folder, save_dir, save_period, tensorboard):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.dataloader = data_loader.get_dataloader()
        self.num_classes = data_loader.get_num_classes()
        self.selfies_alphabet = data_loader.get_selfies_alphabet()
        self.largest_selfie_len = data_loader.get_largest_selfie_len()
        self.int_mol = data_loader.get_int_mol()
        self.dequantized_onehots_min = data_loader.get_dequantized_onehots_min()
        self.dequantized_onehots_max = data_loader.get_dequantized_onehots_max()
        self.classes_breakpoints = data_loader.get_classes_breakpoints()
        self.type_property = data_loader.get_type_property()            

        self.model = Unet1D(
                    dim = unet_dim,
                    dim_mults = (1, 2, 4, 8),
                    num_classes = data_loader.get_num_classes(),
                    channels = unet_channels,
                    cond_drop_prob = unet_cond_drop_prob
                )

        self.diffusion = Diffusion1D(
            self.model,
            seq_length = data_loader.get_seq_length(),
            timesteps = timesteps
        ).to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=lr)

        self.epochs = epochs
        self.results_folder = results_folder
    
    def _train_epoch(self, epoch):
         for step, batch in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                training_mols = batch[0].to(self.device) # normalized from 0 to 1
                mols_classes = batch[1].to(self.device)    # say 10 classes

                loss = self.diffusion(training_mols, classes = mols_classes)

                if step % 100 == 0:
                    print("Loss:", loss.item())

                loss.backward()
                self.optimizer.step()

    def train(self):
        for epoch in range(self.epochs):
            self._train_epoch(epoch)
            nb_mols = 30
            samples_classes = torch.tensor([i // (nb_mols//3) for i in range(nb_mols)]).to(self.device)
            # conditional
            all_continous_mols = torch.squeeze(self.diffusion.sample(samples_classes, cond_scale = 6.))
            recalculate_selfies = utils.mols_continous_to_selfies(all_continous_mols, self.selfies_alphabet, self.largest_selfie_len, self.int_mol, self.dequantized_onehots_min, self.dequantized_onehots_max)
            mols, valid_selfies_list, valid_count = utils.selfies_to_mols(recalculate_selfies)
            print('%.2f' % (valid_count / len(recalculate_selfies)*100),  '% of generated samples are valid molecules.')
            print('Match classes : ', utils.generate_mols_match_classes(mols, samples_classes, self.type_property, self.num_classes, self.classes_breakpoints),'% de réussite')
            img = Chem.Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(200,200), returnPNG=False)
            img.save(str(f'{self.results_folder}mol1D-conditional-{epoch}.png'))