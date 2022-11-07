import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch_geometric.data import DataLoader
from torch_geometric.nn import DataParallel
from argoverse.evaluation.eval_forecasting import get_displacement_errors_and_miss_rate
from argoverse.evaluation.competition_util import generate_forecasting_h5
from argoverse.utils.mpl_plotting_utils import visualize_centerline

from apex import amp
from apex.parallel import DistributedDataParallel

from core.trainer.trainer import Trainer
from core.model.TNT import TNT
from core.optim_schedule import ScheduledOptim
from core.util.viz_utils import show_pred_and_gt
from core.loss import TNTLoss

import pickle
import re


class TNTTrainer(Trainer):
    """
    TNT Trainer, train the TNT with specified hyperparameters and configurations
    """
    def __init__(self,
                 trainset,
                 evalset,
                 testset,
                 batch_size: int = 1,
                 num_workers: int = 1,
                 num_global_graph_layer=1,
                 horizon: int = 30,  #30
                 lr: float = 1e-3,
                 betas=(0.9, 0.999),
                 weight_decay: float = 0.01,
                 warmup_epoch=30,
                 lr_update_freq=5,
                 lr_decay_rate=0.3,
                 aux_loss: bool = False,
                 with_cuda: bool = False,
                 cuda_device=None,
                 multi_gpu=False,
                 enable_log=True,
                 log_freq: int = 2,
                 save_folder: str = "",
                 model_path: str = None,
                 ckpt_path: str = None,
                 verbose: bool = True
                 ):
        """
        trainer class for tnt
        :param trainset: see parent class
        :param evalset: see parent class
        :param testset: see parent class
        :param lr: see parent class
        :param betas: see parent class
        :param weight_decay: see parent class
        :param warmup_steps: see parent class
        :param with_cuda: see parent class
        :param cuda_device: see parent class
        :param multi_gpu: see parent class
        :param log_freq: see parent class
        :param model_path: str, the path to a trained model
        :param ckpt_path: str, the path to a stored checkpoint to be resumed
        :param verbose: see parent class
        """
        super(TNTTrainer, self).__init__(
            trainset=trainset,
            evalset=evalset,
            testset=testset,
            batch_size=batch_size,
            num_workers=num_workers,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            warmup_epoch=warmup_epoch,
            with_cuda=with_cuda,
            cuda_device=cuda_device,
            multi_gpu=multi_gpu,
            enable_log=enable_log,
            log_freq=log_freq,
            save_folder=save_folder,
            verbose=verbose
        )

        # init or load model
        self.horizon = horizon
        self.aux_loss = aux_loss

        self.lambda1 = 0.1
        self.lambda2 = 1.0
        self.lambda3 = 0.1

        # input dim: (20, 8); output dim: (30, 2)
        # model_name = VectorNet
        model_name = TNT
        self.model = model_name(
            self.trainset.num_features if hasattr(self.trainset, 'num_features') else self.testset.num_features,
            self.horizon,
            num_global_graph_layer=num_global_graph_layer,
            with_aux=aux_loss,
            device=self.device
        )
        self.criterion = TNTLoss(
            self.lambda1, self.lambda2, self.lambda3,
            self.model.m, self.model.k, 0.01,
            aux_loss=self.aux_loss,
            device=self.device
        )

        # init optimizer
        self.optim = AdamW(self.model.parameters(), lr=self.lr, betas=self.betas, weight_decay=self.weight_decay)
        self.optm_schedule = ScheduledOptim(
            self.optim,
            self.lr,
            n_warmup_epoch=self.warmup_epoch,
            update_rate=lr_update_freq,
            decay_rate=lr_decay_rate
        )

        # resume from model file or maintain the original
        if model_path:
            self.load(model_path, 'm')

        self.model = self.model.to(self.device)
        if self.multi_gpu:
            self.model = DistributedDataParallel(self.model)
            self.model, self.optimizer = amp.initialize(self.model, self.optim, opt_level="O0")
            if self.verbose and (not self.multi_gpu or (self.multi_gpu and self.cuda_id == 1)):
                print("[TNTTrainer]: Train the mode with multiple GPUs: {} GPUs.".format(int(os.environ['WORLD_SIZE'])))
        else:
            if self.verbose and (not self.multi_gpu or (self.multi_gpu and self.cuda_id == 1)):
                print("[TNTTrainer]: Train the mode with single device on {}.".format(self.device))

        # record the init learning rate
        if not self.multi_gpu or (self.multi_gpu and self.cuda_id == 1):
            self.write_log("LR", self.lr, 0)

        # resume training from ckpt
        if ckpt_path:
            self.load(ckpt_path, 'c')

    def iteration(self, epoch, dataloader):
        training = self.model.training
        avg_loss = 0.0
        num_sample = 0

        data_iter = tqdm(
            enumerate(dataloader),
            desc="Info: Device_{}: {}_Ep_{}: loss: {:.5e}; avg_loss: {:.5e}".format(
                self.cuda_id,
                "train" if training else "eval",
                epoch,
                0.0,
                avg_loss),
            total=len(dataloader),
            bar_format="{l_bar}{r_bar}"
        )

        for i, data in data_iter:
            n_graph = data.num_graphs
            data = data.to(self.device)

            if training:
                self.optm_schedule.zero_grad()
                loss, loss_dict = self.compute_loss(data)

                if self.multi_gpu:
                    with amp.scale_loss(loss, self.optim) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                self.optim.step()

                # writing loss
                if not self.multi_gpu or (self.multi_gpu and self.cuda_id == 1):
                    self.write_log("Train_Loss", loss.detach().item() / n_graph, i + epoch * len(dataloader))
                    self.write_log("Target_Cls_Loss",
                                loss_dict["tar_cls_loss"].detach().item() / n_graph, i + epoch * len(dataloader))
                    self.write_log("Target_Offset_Loss",
                                loss_dict["tar_offset_loss"].detach().item() / n_graph, i + epoch * len(dataloader))
                    self.write_log("Traj_Loss",
                                loss_dict["traj_loss"].detach().item() / n_graph, i + epoch * len(dataloader))
                    self.write_log("Score_Loss",
                                loss_dict["score_loss"].detach().item() / n_graph, i + epoch * len(dataloader))

            else:
                with torch.no_grad():
                    loss, loss_dict = self.compute_loss(data)

                    # writing loss
                    if not self.multi_gpu or (self.multi_gpu and self.cuda_id == 1):
                        self.write_log("Eval_Loss", loss.item() / n_graph, i + epoch * len(dataloader))

            num_sample += n_graph
            avg_loss += loss.detach().item()

            desc_str = "[Info: Device_{}: {}_Ep_{}: loss: {:.5e}; avg_loss: {:.5e}]".format(
                self.cuda_id,
                "train" if training else "eval",
                epoch,
                loss.detach().item() / n_graph,
                avg_loss / num_sample)
            data_iter.set_description(desc=desc_str, refresh=True)

        if training:
            if not self.multi_gpu or (self.multi_gpu and self.cuda_id == 1):
                learning_rate = self.optm_schedule.step_and_update_lr()
                self.write_log("LR", learning_rate, epoch + 1)

        return avg_loss / num_sample

    def compute_loss(self, data):
        n = data.candidate_len_max[0]
        data.y = data.y.view(-1, self.horizon, 2).cumsum(axis=1)
        
        pred, aux_out, aux_gt = self.model(data)

        gt = {
            "target_prob": data.candidate_gt.view(-1, n),
            "offset": data.offset_gt.view(-1, 2),
            "y": data.y.view(-1, self.horizon * 2)
        }
        return self.criterion(pred, gt, aux_out, aux_gt)

    def test(self,
             miss_threshold=2.0,
             compute_metric=False,
             convert_coordinate=False,
             plot=False,
             save_pred=False):
        """
        test the testset,
        :param miss_threshold: float, the threshold for the miss rate, default 2.0m
        :param compute_metric: bool, whether compute the metric
        :param convert_coordinate: bool, True: under original coordinate, False: under the relative coordinate
        :param save_pred: store the prediction or not, store in the Argoverse benchmark format
        """
        self.model.eval()

        forecasted_trajectories, gt_trajectories = {}, {}
        pre_targets = {}

        # k = self.model.k if not self.multi_gpu else self.model.module.k
        k = self.model.k
        # horizon = self.model.horizon if not self.multi_gpu else self.model.module.horizon
        horizon = self.model.horizon

        # debug
        out_dict = {}
        out_cnt = 0

        with torch.no_grad():
            for data in tqdm(self.test_loader):
                batch_size = data.num_graphs
                gt = data.y.unsqueeze(1).view(batch_size, -1, 2).cumsum(axis=1).numpy()

                origs = data.orig.numpy()
                rots = data.rot.numpy()
                seq_ids = data.seq_id.numpy()

                if gt is None:
                    compute_metric = False

                # inference and transform dimension
                if self.multi_gpu:
                    out = self.model.module(data.to(self.device))
                    # out = self.model.inference(data.to(self.device))
                else:
                    out, predcited_targets = self.model.inference_2(data.to(self.device))
                dim_out = len(out.shape)

                # debug
                out_dict[out_cnt] = out.cpu().numpy()
                out_cnt += 1

                pred_y = out.unsqueeze(dim_out).view((batch_size, k, horizon, 2)).cpu().numpy()

                # record the prediction and ground truth
                for batch_id in range(batch_size):
                    seq_id = seq_ids[batch_id]
                    forecasted_trajectories[seq_id] = [self.convert_coord(pred_y_k, origs[batch_id], rots[batch_id])
                                                       if convert_coordinate else pred_y_k
                                                       for pred_y_k in pred_y[batch_id]]
                    gt_trajectories[seq_id] = self.convert_coord(gt[batch_id], origs[batch_id], rots[batch_id]) \
                        if convert_coordinate else gt[batch_id]
                    
                    pre_targets[seq_id] = predcited_targets[batch_id]

        # compute the metric
        if compute_metric:
            metric_results = get_displacement_errors_and_miss_rate(
                forecasted_trajectories,
                gt_trajectories,
                k,
                horizon,
                miss_threshold
            )
            print("[TNTTrainer]: The test result: {};".format(metric_results))

        # plot the result
        # if plot:
        #     fig, ax = plt.subplots()
        #     for key in forecasted_trajectories.keys():
        #         ax.set_xlim(-15, 15)
        #         show_pred_and_gt(ax, gt_trajectories[key], forecasted_trajectories[key])
        #         plt.pause(3)
        #         ax.clear()
        
        # if plot:
        #     overall = {}
        #     overall['gt_trajectories'] = gt_trajectories
        #     overall['forecasted_trajectories'] = forecasted_trajectories
        #     overall['predicted_targets'] = pre_targets
        #     import pickle
        #     with open("/home/kyber/Desktop/unused/TNT/trajectories_targets_smarts.pickle", "wb") as file:
        #         pickle.dump(overall, file, pickle.HIGHEST_PROTOCOL)
        
        # if plot:
        #     for key in forecasted_trajectories.keys():
        #         fig = plt.figure(0, figsize=(8, 7))
        #         fig.clear()
        #         plt.plot(gt_trajectories[key][:, 0], gt_trajectories[key][:, 1], color='orange', alpha=1, linewidth=1, zorder=15)
        #         for i in range(len(forecasted_trajectories[key])):
        #             plt.plot(forecasted_trajectories[key][i][:, 0], forecasted_trajectories[key][i][:, 1], color='green', alpha=1, linewidth=1, zorder=15)
        #             # plt.show()
        #         plt.savefig('/home/kyber/Desktop/unused/TNT/plot_wo_map/{}.png'.format(key))

        if plot:
            self.plot(gt_trajectories, forecasted_trajectories, pre_targets)


        # todo: save the output in argoverse format
        if save_pred:
            for key in forecasted_trajectories.keys():
                forecasted_trajectories[key] = np.asarray(forecasted_trajectories[key])
            generate_forecasting_h5(forecasted_trajectories, self.save_folder)
    
    def plot(self, gt_trajectories, forecasted_trajectories, pre_targets):
        vehicle_data = {}
        vehicle_data['gt_trajectories'] = gt_trajectories
        vehicle_data['forecasted_trajectories'] = forecasted_trajectories
        vehicle_data['predicted_targets'] = pre_targets

        vehicle_ids = []
        for filename in os.listdir('/home/kyber/Desktop/unused/TNT/dataset/interm_data/val_intermediate/raw/'):
            if filename.endswith(".pkl"):
                match = re.search("(.*).pkl", filename)
                assert match is not None
                vehicle_id = match.group(1)
                if vehicle_id not in vehicle_ids:
                    vehicle_ids.append(vehicle_id)

        for id in vehicle_ids:
            with open('/home/kyber/Desktop/unused/TNT/dataset/interm_data/val_intermediate/raw/' + "{}.pkl".format(id), "rb") as f:
                data = pickle.load(f)
                self.visualize_data(data, vehicle_data)
    
    def visualize_data(self, data, vehicle_data):
        """
        visualize the extracted data, and exam the data
        """
        fig = plt.figure(0, figsize=(8, 7))
        fig.clear()

        # visualize the centerlines
        lines_ctrs = data['graph'][0]['ctrs']
        lines_feats = data['graph'][0]['feats']
        lane_idcs = data['graph'][0]['lane_idcs']
        for i in np.unique(lane_idcs):
            line_ctr = lines_ctrs[lane_idcs == i]
            line_feat = lines_feats[lane_idcs == i]
            line_str = (2.0 * line_ctr - line_feat) / 2.0
            line_end = (2.0 * line_ctr[-1, :] + line_feat[-1, :]) / 2.0
            line = np.vstack([line_str, line_end.reshape(-1, 2)])
            visualize_centerline(line)

        # visualize the trajectory
        past_trajs = data['feats'][0][:, :, :2]
        has_obss = data['has_obss'][0]
        
        obs = past_trajs[0]
        gt_trj = vehicle_data['gt_trajectories'][int(data['seq_id'][0])]
        pred_trjs = vehicle_data['forecasted_trajectories'][int(data['seq_id'][0])]
        predicted_targets = vehicle_data['predicted_targets'][int(data['seq_id'][0])]

        plt.plot(obs[:, 0], obs[:, 1], color='red', alpha=1, linewidth=1, zorder=15)
        plt.plot(gt_trj[:, 0], gt_trj[:, 1], color='orange', alpha=1, linewidth=1, zorder=15)
        for i in range(len(pred_trjs)):
            plt.plot(pred_trjs[i][:, 0], pred_trjs[i][:, 1], color='green', alpha=1, linewidth=1, zorder=15)
        
        # x = predicted_targets[:,0]
        # y = predicted_targets[:,1]
        # plt.scatter(x,y)

        plt.xlabel("Map X")
        plt.ylabel("Map Y")
        plt.axis("off")

        plt.savefig('/home/kyber/Desktop/plot_check/{}.png'.format(data['seq_id'][0]), dpi=600)


    def predict(self, data, convert_coordinate=False):
        k = 6
        horizon = 30

        with torch.no_grad():
            orig = data.orig.numpy()
            rot = data.rot.numpy()

            out = self.model.inference(data.to(self.device))
            dim_out = len(out.shape)
            pred_y = out.unsqueeze(dim_out).view((1, k, horizon, 2)).cpu().numpy()

            forecasted_trajectories = [self.convert_coord(pred_y_k, orig, rot)
                                                if convert_coordinate else pred_y_k
                                        for pred_y_k in pred_y[0]]
        return forecasted_trajectories

    # function to convert the coordinates of trajectories from relative to world
    def convert_coord(self, traj, orig, rot):
        traj_converted = np.matmul(np.linalg.inv(rot), traj.T).T + orig.reshape(-1, 2)
        return traj_converted

