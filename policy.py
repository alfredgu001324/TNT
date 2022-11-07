import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
from typing import Any, Dict
# from smarts.env.wrappers.format_action import FormatAction
# from smarts.env.wrappers.format_obs import FormatObs
# from smarts.core.controllers import ActionSpaceType
from utility import *
from core.trainer.tnt_trainer import TNTTrainer
from core.dataloader.argoverse_loader_v2 import ArgoverseInDisk

class BasePolicy:
    def act(self, obs: Dict[str, Any]):
        """Act function to be implemented by user.
        Args:
            obs (Dict[str, Any]): A dictionary of observation for each ego agent step.
        Returns:
            Dict[str, Any]: A dictionary of actions for each ego agent.
        """
        raise NotImplementedError


# def submitted_wrappers():
#     """Return environment wrappers for wrapping the evaluation environment.
#     Each wrapper is of the form: Callable[[env], env]. Use of wrappers is
#     optional. If wrappers are not used, return empty list [].
#     Returns:
#         List[wrappers]: List of wrappers. Default is empty list [].
#     """

#     # Insert wrappers here, if any.
#     wrappers = [
#         FormatObs,
#         lambda env: FormatAction(env=env, space=ActionSpaceType["TargetPose"]),
#     ]

#     return wrappers


class Policy(BasePolicy):
    """Policy class to be submitted by the user. This class will be loaded
    and tested during evaluation."""

    def __init__(self):
        """All policy initialization matters, including loading of model, is
        performed here. To be implemented by the user.
        """
        # init trainer
        # self.model = TNT(10)
        # self.model.load_state_dict(torch.load('/home/kyber/Desktop/unused/TNT/run/tnt/10-19-18-52/best_TNT.pth', map_location=self.model.device))
        # with open('/home/kyber/Desktop/unused/TNT/dataset/interm_data/val_intermediate/raw/features_147.pkl', "rb") as f:
        #     single_vehicle_data = pd.read_pickle(f)
        # graph_data = ArgoversetoGraph(raw_data=single_vehicle_data).process()
        graph_data = ArgoverseInDisk('/home/kyber/Desktop/unused/TNT/dataset/interm_data/val_intermediate')
        self.trainer = TNTTrainer(
            trainset=graph_data,
            evalset=graph_data,
            testset=graph_data,
            batch_size=128,
            num_workers=16,
            aux_loss=True,
            enable_log=False,
            model_path='/home/kyber/Desktop/unused/TNT/run/tnt/10-19-18-52/best_TNT.pth'
        )
        self.vehicle_data = {}
    
    def act(self, obs: Dict[str, Any]):
        """Act function to be implemented by user.
        Args:
            obs (Dict[str, Any]): A dictionary of observation for each ego agent step.
        Returns:
            Dict[str, Any]: A dictionary of actions for each ego agent.
        """

        # Use saved model to predict multi-agent action output given multi-agent SMARTS observation input.
        wrapped_act = {}
        for agent_id, agent_obs in obs.items():
            # self.vehicle_data.update({agent_id: s2a(self.vehicle_data[agent_id], agent_obs)})
            with open('/home/kyber/Desktop/unused/TNT/dataset/interm_data/val_intermediate/raw/features_7.pkl', "rb") as f:
                single_vehicle_data = pd.read_pickle(f)
            self.vehicle_data.update({agent_id: single_vehicle_data})

            graph_data = ArgoversetoGraph(raw_data=self.vehicle_data[agent_id]).process()

            # forecasted_trajectories = predict(self.model, graph_data, convert_coordinate=False) # shape(forecatsed_tr) = 6, 30, 2
            forecasted_trajectories = self.trainer.predict(graph_data, convert_coordinate=False)
            breakpoint()
            next_position = forecasted_trajectories[0][0]
            current_x = agent_obs["ego"]["pos"][0]
            current_y = agent_obs["ego"]["pos"][1]
            current_heading = agent_obs["ego"]["heading"]

            dheading = get_dheading(next_position[0], next_position[1], current_x, current_y, current_heading)
            target_pose = np.array([next_position[0], next_position[1], dheading + current_heading, 0.1], dtype=object)
            wrapped_act.update({agent_id: target_pose})

        return wrapped_act

if __name__ == "__main__":
    policy = Policy()
    obs = {'1': 'aaaaa'}
    policy.act(obs)

