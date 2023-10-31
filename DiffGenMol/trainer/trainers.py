from accelerate import Accelerator
from ema_pytorch import EMA
import math
import torch
import utils
from diffusion.diffusion_1d import Diffusion1D
from logger import TensorboardWriter
from model.unet_1d import Unet1D
from model import metrics
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms
from tqdm.auto import tqdm

def exists(x):
    return x is not None

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

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
                timesteps = timesteps
            )
        else:
            self.model = model

        self.channels = self.data_loader.get_num_classes()

        # sampling and training hyperparameters

        assert has_int_squareroot(self.data_loader.get_nb_mols()), 'number of samples must have an integer square root'
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
        self.train_metrics = utils.MetricTracker('loss', 'syntax_validity_uc', 'mol_validity_uc','uniqueness_uc',
                                                 'novelty_uc','KLdiv_uc','syntax_validity_c', 'validity_c',
                                                 'uniqueness_c','novelty_c','KLdiv_c','classes_match', writer=self.writer)

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
                self.train_metrics.update('loss', total_loss)

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()
                    milestone = self.step // self.eval_and_sample_every
                    if self.step != 0 and self.step % self.eval_and_sample_every == 0:
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

        # metrics
        self.train_metrics.update('syntax_validity_uc', metrics.accuracy_valid_conversion_selfies_to_smiles(samples_selfies))
        self.train_metrics.update('mol_validity_uc', metrics.validity_score(samples_smiles))
        self.train_metrics.update('uniqueness_uc', metrics.uniqueness_score(samples_smiles))
        self.train_metrics.update('novelty_uc', metrics.novelty_score(samples_smiles, self.data_loader.get_train_smiles()))
        self.train_metrics.update('KLdiv_uc', metrics.KLdiv_score(samples_smiles, self.data_loader.get_train_smiles()))
        # image
        self.writer.add_image('mol_uc', transforms.ToTensor()(Chem.Draw.MolToImage(samples_mols[0],size=(300,300))))

        # sample
        df = pd.DataFrame(samples_selfies, columns=["selfies"])
        df.to_csv(f'{self.result_dir}/smiles-uc-{milestone}.csv', index=False)
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
        
        # metrics
        self.train_metrics.update('syntax_validity_c', metrics.accuracy_valid_conversion_selfies_to_smiles(samples_selfies))
        self.train_metrics.update('validity_c', metrics.validity_score(samples_smiles))
        self.train_metrics.update('uniqueness_c', metrics.uniqueness_score(samples_smiles))
        self.train_metrics.update('novelty_c', metrics.novelty_score(samples_smiles, self.data_loader.get_train_smiles()))
        self.train_metrics.update('KLdiv_c', metrics.KLdiv_score(samples_smiles, self.data_loader.get_train_smiles()))
        self.train_metrics.update('classes_match', metrics.accuracy_match_classes(samples_mols, samples_classes, self.data_loader.get_type_property(), self.data_loader.get_num_classes(), self.data_loader.get_classes_breakpoints()))
        # image
        self.writer.add_image('mol_c', transforms.ToTensor()(Chem.Draw.MolsToGridImage(samples_mols, molsPerRow=_num_samples//self.data_loader.get_num_classes(), subImgSize=(300,300))))

        # sample
        df = pd.DataFrame(samples_selfies, columns=["selfies"])
        df.to_csv(f'{self.result_dir}/smiles-c-{milestone}.csv', index=False)
        df = pd.DataFrame(samples_smiles, columns=["smiles"])
        df.to_csv(f'{self.result_dir}/smiles-c-{milestone}.csv', index=False)
