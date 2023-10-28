import torch
import utils
from diffusion.diffusion_1d import Diffusion1D
from logger import TensorboardWriter
from model.unet_1d import Unet1D
from model import metrics
from rdkit import Chem
from rdkit.Chem import Draw
from torch.optim import Adam
from torchvision import transforms

class Trainer1D():
    """
    Trainer 1D class
    """
    def __init__(self, data_loader, timesteps, epochs, lr, unet_dim, unet_channels, unet_cond_drop_prob, results_folder, config, save_dir, save_period, tensorboard):
        self.config = config
        self.logger = config.get_logger('train')
        self.adam_betas = (0.9, 0.99)
        # prepare for (multi-device) GPU training
        device, device_ids = utils.prepare_device(self.config['n_gpu'])

        self.device = device
        self.data_loader = data_loader
        self.len_epoch = len(self.data_loader.get_dataloader())

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

        self.optimizer = Adam(self.model.parameters(), lr=lr, betas=self.adam_betas)

        self.start_epoch = 1
        self.epochs = epochs
        self.results_folder = results_folder

        # setup visualization writer instance                
        self.writer = TensorboardWriter(config.log_dir, self.logger, self.config['trainer']['args']['tensorboard'])
        self.train_metrics = utils.MetricTracker('loss', 'uncond_validity','uncond_uniqueness','uncond_novelty','uncond_KLdiv','cond_validity','cond_uniqueness','cond_novelty','cond_KLdiv','classes_match', writer=self.writer)
    
    def _train_epoch(self, epoch):
        self.train_metrics.reset()
        for step, batch in enumerate(self.data_loader.get_dataloader()):
            self.optimizer.zero_grad()
            training_mols = batch[0].to(self.device) # normalized from 0 to 1
            mols_classes = batch[1].to(self.device)    # say 10 classes

            loss = self.diffusion(training_mols, classes = mols_classes)
            loss.backward()
            self.optimizer.step()

            if step % 50 == 0:
                print("Loss:", loss.item())

            writer_step = (epoch - 1) * self.len_epoch + step

            self.writer.set_step(writer_step)
            self.train_metrics.update('loss', loss.item())

            if writer_step % 250 == 0:
                with torch.no_grad():
                    # unconditional
                    # image
                    nb_mols = 1 * self.data_loader.get_num_classes()
                    samples_classes = torch.tensor([i // (nb_mols//self.data_loader.get_num_classes()) for i in range(nb_mols)]).to(self.device)
                    
                    samples_continous_mols = torch.squeeze(self.diffusion.sample(samples_classes, cond_scale = 0))
                    samples_selfies = utils.continous_mols_to_selfies(samples_continous_mols, self.data_loader.get_selfies_alphabet(), self.data_loader.get_largest_selfie_len(), 
                                                                        self.data_loader.get_int_mol(), self.data_loader.get_dequantized_onehots_min(), self.data_loader.get_dequantized_onehots_max())
                    samples_mols, _, _ = utils.selfies_to_mols(samples_selfies)
                    self.writer.add_image('gen_mol_uncond', transforms.ToTensor()(Chem.Draw.MolToImage(samples_mols[0],size=(300,300))))

                    # metrics
                    nb_mols = 10 * self.data_loader.get_num_classes()
                    samples_classes = torch.tensor([i // (nb_mols//self.data_loader.get_num_classes()) for i in range(nb_mols)]).to(self.device)
                    samples_continous_mols = torch.squeeze(self.diffusion.sample(samples_classes, cond_scale = 0))
                    samples_selfies = utils.continous_mols_to_selfies(samples_continous_mols, self.data_loader.get_selfies_alphabet(), self.data_loader.get_largest_selfie_len(), 
                                                                        self.data_loader.get_int_mol(), self.data_loader.get_dequantized_onehots_min(), self.data_loader.get_dequantized_onehots_max())
                    samples_mols, _, _ = utils.selfies_to_mols(samples_selfies)
                    samples_smiles = utils.mols_to_smiles(samples_mols)

                    self.train_metrics.update('uncond_validity', metrics.validity_score(samples_smiles))
                    self.train_metrics.update('uncond_uniqueness', metrics.uniqueness_score(samples_smiles))
                    self.train_metrics.update('uncond_novelty', metrics.novelty_score(samples_smiles, self.data_loader.get_train_smiles()))
                    self.train_metrics.update('uncond_KLdiv', metrics.KLdiv_score(samples_smiles, self.data_loader.get_train_smiles()))
                    #self.train_metrics.update('uncond_similarity', metrics.top3_tanimoto_similarity_score(samples_mols, self.data_loader.get_train_mols()))
                    
                    # conditional
                    # image
                    nb_mols = 3 * self.data_loader.get_num_classes()
                    samples_classes = torch.tensor([i // (nb_mols//self.data_loader.get_num_classes()) for i in range(nb_mols)]).to(self.device)
                    samples_continous_mols = torch.squeeze(self.diffusion.sample(samples_classes, cond_scale = 6.))
                    samples_selfies = utils.continous_mols_to_selfies(samples_continous_mols, self.data_loader.get_selfies_alphabet(), self.data_loader.get_largest_selfie_len(), 
                                                                        self.data_loader.get_int_mol(), self.data_loader.get_dequantized_onehots_min(), self.data_loader.get_dequantized_onehots_max())
                    samples_mols, _, _ = utils.selfies_to_mols(samples_selfies)
                    
                    self.writer.add_image('gen_mols_cond', transforms.ToTensor()(Chem.Draw.MolsToGridImage(samples_mols, molsPerRow=3, subImgSize=(200,200))))

                    # metrics cond
                    nb_mols = 10 * self.data_loader.get_num_classes()
                    samples_classes = torch.tensor([i // (nb_mols//self.data_loader.get_num_classes()) for i in range(nb_mols)]).to(self.device)
                    samples_continous_mols = torch.squeeze(self.diffusion.sample(samples_classes, cond_scale = 6.))
                    samples_selfies = utils.continous_mols_to_selfies(samples_continous_mols, self.data_loader.get_selfies_alphabet(), self.data_loader.get_largest_selfie_len(), 
                                                                        self.data_loader.get_int_mol(), self.data_loader.get_dequantized_onehots_min(), self.data_loader.get_dequantized_onehots_max())
                    samples_mols, _, _ = utils.selfies_to_mols(samples_selfies)
                    samples_smiles = utils.mols_to_smiles(samples_mols)

                    self.train_metrics.update('cond_validity', metrics.validity_score(samples_smiles))
                    self.train_metrics.update('cond_uniqueness', metrics.uniqueness_score(samples_smiles))
                    self.train_metrics.update('cond_novelty', metrics.novelty_score(samples_smiles, self.data_loader.get_train_smiles()))
                    self.train_metrics.update('cond_KLdiv', metrics.KLdiv_score(samples_smiles, self.data_loader.get_train_smiles()))
                    #print(f'          KLdiv cond. score: {round(metrics.KLdiv_score_cond(samples_smiles, samples_classes, self.data_loader.get_train_smiles(), self.data_loader.get_train_classes(), self.data_loader.get_num_classes()), 3)}')               
                    #self.train_metrics.update('classes_similarity', metrics.top3_tanimoto_similarity_score_cond(samples_mols, samples_classes, self.data_loader.get_train_mols(), self.data_loader.get_train_classes(), self.data_loader.get_num_classes()))
                    self.train_metrics.update('classes_match', metrics.accuracy_match_classes(samples_mols, samples_classes, self.data_loader.get_type_property(), self.data_loader.get_num_classes(), self.data_loader.get_classes_breakpoints()))
                    
        return
        

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            self._train_epoch(epoch)
            result = self.train_metrics.result()
            # save logged informations into log dict
            log = {'epoch': epoch}
            log.update(result)

            # print logged informations to the screen
            for key, value in log.items():
                self.logger.info('    {:15s}: {}'.format(str(key), value))
