from accelerate import Accelerator
from ema_pytorch import EMA
import torch
import utils
from diffusion.diffusion_1d import Diffusion1D
from logger import TensorboardWriter
from model.unet_1d import Unet1D
from model import metrics
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
from tqdm.auto import tqdm

def exists(x):
    return x is not None

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

class Trainer1D(object):
    def __init__(
        self,
        data_loader, 
        timesteps, 
        unet_dim, 
        unet_channels, 
        unet_cond_drop_prob,
        objective,
        beta_schedule, 
        tensorboard,
        config,
        *,
        model = None,
        gradient_accumulate_every = 2,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        eval_and_sample_every = 1000,
        save_model_every = 2000,
        num_samples = 20,
        amp = True,
        mixed_precision_type = 'fp16',
        split_batches = True,
        max_grad_norm = 1.
    ):
        super().__init__()

        self.config = config
        self.logger = config.get_logger('train')

        self.data_loader = data_loader

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model

        if model is None:
            unet = Unet1D(
                        dim = unet_dim,
                        dim_mults = (1, 2, 4, 8),
                        num_classes = self.data_loader.get_num_classes(),
                        channels = unet_channels,
                        cond_drop_prob = unet_cond_drop_prob
                        )
            
            self.model = Diffusion1D(
                unet,
                seq_length = self.data_loader.get_seq_length(),
                timesteps = timesteps,
                objective = objective,
                beta_schedule = beta_schedule
            )
        else:
            self.model = model

        self.channels = self.data_loader.get_num_classes()

        # sampling and training hyperparameters

        self.num_samples = num_samples
        self.eval_and_sample_every = eval_and_sample_every
        self.save_model_every = save_model_every

        self.batch_size = self.data_loader.get_batch_size()
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_grad_norm = max_grad_norm

        self.train_num_steps = train_num_steps

        # dataset and dataloader

        dl = self.accelerator.prepare(self.data_loader.get_dataloader())
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(self.model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(self.model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)

        self.result_dir = self.config.result_dir
        self.model_dir = self.config.model_dir
        self.log_dir = self.config.log_dir

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # setup visualization writer instance                
        self.writer = TensorboardWriter(self.log_dir, self.logger, tensorboard)
        self.train_metrics = utils.MetricTracker('0_loss', 'A_UC_01_syntax_validity_samples', 'A_UC_02_mol_validity_samples',
                                                 'A_UC_03_uniqueness_samples', 'A_UC_04_novelty_samples','A_UC_05_KLdiv_samples',
                                                 'A_UC_06a_distr_weight_train', 'A_UC_06b_distr_weight_samples','A_UC_06c_wasserstein_distance_weight',
                                                 'A_UC_07a_distr_logp_train', 'A_UC_07b_distr_logp_samples','A_UC_07c_wasserstein_distance_logp',
                                                 'A_UC_08a_distr_qed_train', 'A_UC_08b_distr_qed_samples','A_UC_08c_wasserstein_distance_qed',
                                                 'A_UC_09a_distr_sas_train', 'A_UC_09b_distr_sas_samples','A_UC_09c_wasserstein_distance_sas',
                                                 'A_UC_10_wasserstein_distance_sum_props',
                                                 'B_C_01_syntax_validity_samples', 'B_C_02_mol_validity_samples',
                                                 'B_C_03_uniqueness_samples','B_C_04_novelty_samples','B_C_05_KLdiv_samples',
                                                 'B_C_06a_distr_prop_train', 'B_C_06b_distr_prop_samples','B_C_06c_wasserstein_distance_prop',
                                                 'B_C_07a_distr_classe_1_train', 'B_C_07b_distr_classe_1_samples','B_C_07c_wasserstein_distance_classe_1',
                                                 'B_C_08a_distr_classe_2_train', 'B_C_08b_distr_classe_2_samples','B_C_08c_wasserstein_distance_classe_2', 
                                                 'B_C_09a_distr_classe_3_train', 'B_C_09b_distr_classe_3_samples','B_C_09c_wasserstein_distance_classe_3',
                                                 'B_C_10_wasserstein_distance_sum_classes',
                                                 'B_C_11_classes_match', writer=self.writer)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
            'version': '1.0'
        }

        torch.save(data, str(self.model_dir / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.model_dir / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl)
                    train_value = data['continous_selfies'].to(device) # normalized from 0 to 1
                    train_classe = data['classe'].to(device)    # say 10 classes

                    with self.accelerator.autocast():
                        loss = self.model(train_value, classes = train_classe)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')
                self.writer.set_step(self.step)
                self.train_metrics.update('0_loss', total_loss)

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()
                    milestone = self.step // self.eval_and_sample_every
                    if self.step == 100 or self.step % self.eval_and_sample_every == 0:
                        with torch.no_grad():
                            self.eval_and_sample(milestone)
                            result = self.train_metrics.result()
                            # save logged informations into log dict
                            log = {'milestone': milestone, 'step': self.step}
                            log.update(result)
                            # print logged informations to the screen
                            for key, value in log.items():
                                self.logger.info('    {:15s}: {}'.format(str(key), value))
                    if self.step != 0 and self.step % self.save_model_every == 0:
                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')
    
    def eval_and_sample(self, milestone):
        # unconditional
        _num_samples = (self.num_samples // self.data_loader.get_num_classes()) * self.data_loader.get_num_classes()
        samples_classes = torch.tensor([i // (_num_samples//self.data_loader.get_num_classes()) for i in range(_num_samples)]).to(self.device)
        samples_continous_mols = torch.squeeze(self.ema.ema_model.sample(samples_classes, cond_scale = 0))
        samples_selfies = utils.continous_mols_to_selfies(samples_continous_mols, self.data_loader.get_selfies_alphabet(), self.data_loader.get_largest_selfie_len(), 
                                                            self.data_loader.get_int_mol())
        samples_mols, _, _ = utils.selfies_to_mols(samples_selfies)
        samples_smiles = utils.mols_to_smiles(samples_mols)
        samples_smiles = utils.canonicalize_smiles(samples_smiles)

        # metrics
        self.train_metrics.update('A_UC_01_syntax_validity_samples', metrics.accuracy_valid_conversion_selfies_to_smiles(samples_selfies))
        self.train_metrics.update('A_UC_02_mol_validity_samples', metrics.validity_score(samples_smiles))
        self.train_metrics.update('A_UC_03_uniqueness_samples', metrics.uniqueness_score(samples_smiles))
        self.train_metrics.update('A_UC_04_novelty_samples', metrics.novelty_score(samples_smiles, self.data_loader.get_train_smiles()))
        self.train_metrics.update('A_UC_05_KLdiv_samples', metrics.KLdiv_score(samples_smiles, self.data_loader.get_train_smiles()))
        
        # only first eval
        if self.step == self.eval_and_sample_every:
            self.writer.add_histogram('A_UC_06a_distr_weight_train',torch.FloatTensor(self.data_loader.get_prop_weight()))
            self.writer.add_histogram('A_UC_07a_distr_logp_train',torch.FloatTensor(self.data_loader.get_prop_logp()))
            self.writer.add_histogram('A_UC_08a_distr_qed_train',torch.FloatTensor(self.data_loader.get_prop_qed()))
            self.writer.add_histogram('A_UC_09a_distr_sas_train',torch.FloatTensor(self.data_loader.get_prop_sas()))

        self.writer.add_histogram('A_UC_06b_distr_weight_samples',torch.FloatTensor(utils.get_smiles_properties(samples_smiles,'Weight')))
        w_distance_weight = metrics.wasserstein_distance_score(utils.get_smiles_properties(samples_smiles,'Weight'), self.data_loader.get_prop_weight())
        self.train_metrics.update('A_UC_06c_wasserstein_distance_weight', w_distance_weight)
        self.writer.add_histogram('A_UC_07b_distr_logp_samples',torch.FloatTensor(utils.get_smiles_properties(samples_smiles,'LogP')))
        w_distance_logp = metrics.wasserstein_distance_score(utils.get_smiles_properties(samples_smiles,'LogP'), self.data_loader.get_prop_logp())
        self.train_metrics.update('A_UC_07c_wasserstein_distance_logp', w_distance_logp)
        self.writer.add_histogram('A_UC_08b_distr_qed_samples',torch.FloatTensor(utils.get_smiles_properties(samples_smiles,'QED')))
        w_distance_qed = metrics.wasserstein_distance_score(utils.get_smiles_properties(samples_smiles,'QED'), self.data_loader.get_prop_qed())
        self.train_metrics.update('A_UC_08c_wasserstein_distance_qed', w_distance_qed)
        self.writer.add_histogram('A_UC_09b_distr_sas_samples',torch.FloatTensor(utils.get_smiles_properties(samples_smiles,'SAS')))
        w_distance_sas = metrics.wasserstein_distance_score(utils.get_smiles_properties(samples_smiles,'SAS'), self.data_loader.get_prop_sas())
        self.train_metrics.update('A_UC_09c_wasserstein_distance_sas',w_distance_sas)
        self.train_metrics.update('A_UC_10_wasserstein_distance_sum_props',w_distance_weight+w_distance_logp+w_distance_qed+w_distance_sas)      
        # image
        self.writer.add_image('A_UC_11_mol_sample', transforms.ToTensor()(Chem.Draw.MolToImage(samples_mols[0],size=(300,300))))

        # sample
        df = pd.DataFrame(samples_selfies, columns=["selfies"])
        df.to_csv(f'{self.result_dir}/selfies-uc-{milestone}.csv', index=False)
        df = pd.DataFrame(samples_smiles, columns=["smiles"])
        df.to_csv(f'{self.result_dir}/smiles-uc-{milestone}.csv', index=False)

        # conditional
        _num_samples = (self.num_samples // self.data_loader.get_num_classes()) * self.data_loader.get_num_classes()
        samples_classes = torch.tensor([i // (_num_samples//self.data_loader.get_num_classes()) for i in range(_num_samples)]).to(self.device)
        samples_continous_mols = torch.squeeze(self.ema.ema_model.sample(samples_classes, cond_scale = 6.))
        samples_selfies = utils.continous_mols_to_selfies(samples_continous_mols, self.data_loader.get_selfies_alphabet(), self.data_loader.get_largest_selfie_len(), 
                                                            self.data_loader.get_int_mol())
        samples_mols, _, _ = utils.selfies_to_mols(samples_selfies)
        samples_smiles = utils.mols_to_smiles(samples_mols)
        samples_smiles = utils.canonicalize_smiles(samples_smiles)
        
        # metrics
        self.train_metrics.update('B_C_01_syntax_validity_samples', metrics.accuracy_valid_conversion_selfies_to_smiles(samples_selfies))
        self.train_metrics.update('B_C_02_mol_validity_samples', metrics.validity_score(samples_smiles))
        self.train_metrics.update('B_C_03_uniqueness_samples', metrics.uniqueness_score(samples_smiles))
        self.train_metrics.update('B_C_04_novelty_samples', metrics.novelty_score(samples_smiles, self.data_loader.get_train_smiles()))
        self.train_metrics.update('B_C_05_KLdiv_samples', metrics.KLdiv_score(samples_smiles, self.data_loader.get_train_smiles()))
        
        # only first eval
        if self.step == self.eval_and_sample_every:
            self.writer.add_histogram('B_C_06a_distr_prop_train',torch.FloatTensor(self.data_loader.get_train_prop()))
            self.writer.add_histogram('B_C_07a_distr_classe_1_train',torch.FloatTensor(utils.get_values_by_classe(self.data_loader.get_train_prop(), self.data_loader.get_train_classes(), 0)))
            self.writer.add_histogram('B_C_08a_distr_classe_2_train',torch.FloatTensor(utils.get_values_by_classe(self.data_loader.get_train_prop(), self.data_loader.get_train_classes(), 1)))
            self.writer.add_histogram('B_C_09a_distr_classe_3_train',torch.FloatTensor(utils.get_values_by_classe(self.data_loader.get_train_prop(), self.data_loader.get_train_classes(), 2)))

        prop_samples = utils.get_smiles_properties(samples_smiles,self.data_loader.get_type_property())
        self.writer.add_histogram('B_C_06b_distr_prop_samples',torch.FloatTensor(prop_samples))
        w_distance = metrics.wasserstein_distance_score(utils.get_smiles_properties(samples_smiles,self.data_loader.get_type_property()), self.data_loader.get_train_prop())
        self.train_metrics.update('B_C_06c_wasserstein_distance_prop', w_distance)
        self.writer.add_histogram('B_C_07b_distr_classe_1_samples',torch.FloatTensor(utils.get_smiles_properties(utils.get_values_by_classe(samples_smiles, samples_classes, 0),self.data_loader.get_type_property())))
        classe_1_w_distance = metrics.wasserstein_distance_score(utils.get_smiles_properties(utils.get_values_by_classe(samples_smiles, samples_classes, 0),self.data_loader.get_type_property()), utils.get_values_by_classe(self.data_loader.get_train_prop(), self.data_loader.get_train_classes(), 0))
        self.train_metrics.update('B_C_07c_wasserstein_distance_classe_1', classe_1_w_distance)
        self.writer.add_histogram('B_C_08b_distr_classe_2_samples',torch.FloatTensor(utils.get_smiles_properties(utils.get_values_by_classe(samples_smiles, samples_classes, 1),self.data_loader.get_type_property())))
        classe_2_w_distance = metrics.wasserstein_distance_score(utils.get_smiles_properties(utils.get_values_by_classe(samples_smiles, samples_classes, 1),self.data_loader.get_type_property()), utils.get_values_by_classe(self.data_loader.get_train_prop(), self.data_loader.get_train_classes(), 1))
        self.train_metrics.update('B_C_08c_wasserstein_distance_classe_2', classe_2_w_distance)
        self.writer.add_histogram('B_C_09b_distr_classe_3_samples',torch.FloatTensor(utils.get_smiles_properties(utils.get_values_by_classe(samples_smiles, samples_classes, 2),self.data_loader.get_type_property())))
        classe_3_w_distance = metrics.wasserstein_distance_score(utils.get_smiles_properties(utils.get_values_by_classe(samples_smiles, samples_classes, 2),self.data_loader.get_type_property()), utils.get_values_by_classe(self.data_loader.get_train_prop(), self.data_loader.get_train_classes(), 2))
        self.train_metrics.update('B_C_09c_wasserstein_distance_classe_3', classe_3_w_distance)
        self.train_metrics.update('B_C_10_wasserstein_distance_sum_classes', classe_1_w_distance+classe_2_w_distance+classe_3_w_distance)

        self.train_metrics.update('B_C_11_classes_match', metrics.accuracy_match_classes(samples_mols, samples_classes, self.data_loader.get_type_property(), self.data_loader.get_num_classes(), self.data_loader.get_classes_breakpoints()))
        # image
        self.writer.add_image('B_C_12_mol_samples', transforms.ToTensor()(Chem.Draw.MolsToGridImage(samples_mols[:20], molsPerRow=20//self.data_loader.get_num_classes(), subImgSize=(300,300))))

        # sample
        d = {'selfies':samples_selfies,self.data_loader.get_type_property():prop_samples,'classe':samples_classes.cpu()}
        df = pd.DataFrame(d)
        df.to_csv(f'{self.result_dir}/selfies-c-{milestone}.csv', index=False)
        d = {'smiles':samples_smiles,self.data_loader.get_type_property():prop_samples,'classe':samples_classes.cpu()}
        df = pd.DataFrame(d)
        df.to_csv(f'{self.result_dir}/smiles-c-{milestone}.csv', index=False)
