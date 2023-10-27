import torch
import utils
from diffusion.diffusion_1d import Diffusion1D
from model.unet_1d import Unet1D
from model import metrics
from rdkit import Chem
from rdkit.Chem import Draw
from torch.optim import Adam

class Trainer1D():
    """
    Trainer 1D class
    """
    def __init__(self, data_loader, timesteps, epochs, lr, unet_dim, unet_channels, unet_cond_drop_prob, results_folder, config, save_dir, save_period, tensorboard):
        self.config = config

        # prepare for (multi-device) GPU training
        device, device_ids = utils.prepare_device(self.config['n_gpu'])

        self.device = device
        self.data_loader = data_loader

        self.model = Unet1D(
                    dim = unet_dim,
                    dim_mults = (1, 2, 4, 8),
                    num_classes = data_loader.get_num_classes(),
                    channels = unet_channels,
                    cond_drop_prob = unet_cond_drop_prob
                ).to(self.device)
   
        if len(device_ids) > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)

        self.diffusion = Diffusion1D(
            self.model,
            seq_length = data_loader.get_seq_length(),
            timesteps = timesteps
        ).to(self.device)

        self.optimizer = Adam(self.model.parameters(), lr=lr)

        self.epochs = epochs
        self.results_folder = results_folder
    
    def _train_epoch(self, epoch):
         for step, batch in enumerate(self.data_loader.get_dataloader()):
            self.optimizer.zero_grad()
            training_mols = batch[0].to(self.device) # normalized from 0 to 1
            mols_classes = batch[1].to(self.device)    # say 10 classes

            loss = self.diffusion(training_mols, classes = mols_classes)

            if step % 50 == 0:
                print("Loss:", loss.item())

            loss.backward()
            self.optimizer.step()

    def train(self):
        for epoch in range(self.epochs):
            self._train_epoch(epoch)
            with torch.no_grad():
                nb_mols = 3 * self.data_loader.get_num_classes()
                samples_classes = torch.tensor([i // (nb_mols//self.data_loader.get_num_classes()) for i in range(nb_mols)]).to(self.device)
                # conditional
                samples_continous_mols = torch.squeeze(self.diffusion.sample(samples_classes, cond_scale = 6.))
                samples_selfies = utils.continous_mols_to_selfies(samples_continous_mols, self.data_loader.get_selfies_alphabet(), self.data_loader.get_largest_selfie_len(), 
                                                                    self.data_loader.get_int_mol(), self.data_loader.get_dequantized_onehots_min(), self.data_loader.get_dequantized_onehots_max())
                samples_mols, _, _ = utils.selfies_to_mols(samples_selfies)
                samples_smiles = utils.mols_to_smiles(samples_mols)

                print(f'             Validity score: {round(metrics.validity_score(samples_smiles), 3)}')
                print(f'           Uniqueness score: {round(metrics.uniqueness_score(samples_smiles), 3)}')
                print(f'              Novelty score: {round(metrics.novelty_score(samples_smiles, self.data_loader.get_train_smiles()), 3)}')
                print(f'                KLdiv score: {round(metrics.KLdiv_score(samples_smiles, self.data_loader.get_train_smiles()), 3)}')
                print(f'      Top3 similarity score: {round(metrics.top3_tanimoto_similarity_score(samples_mols, self.data_loader.get_train_mols()), 3)}')
                #print(f'          KLdiv cond. score: {round(metrics.KLdiv_score_cond(samples_smiles, samples_classes, self.data_loader.get_train_smiles(), self.data_loader.get_train_classes(), self.data_loader.get_num_classes()), 3)}')
                print(f'Top3 similarity cond. score: {round(metrics.top3_tanimoto_similarity_score_cond(samples_mols, samples_classes, self.data_loader.get_train_mols(), self.data_loader.get_train_classes(), self.data_loader.get_num_classes()), 3)}')
                print(f'                Cond. match: {round((metrics.accuracy_match_classes(samples_mols, samples_classes, self.data_loader.get_type_property(), self.data_loader.get_num_classes(), self.data_loader.get_classes_breakpoints())*100), 3)}%')
                img = Chem.Draw.MolsToGridImage(samples_mols, molsPerRow=3, subImgSize=(200,200), returnPNG=False)
                img.save(str(f'{self.results_folder}mol1D-conditional-{epoch}.png'))