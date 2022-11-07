import math
import os
import os.path as osp
import numpy as np

import gc
from copy import copy
from pandas import DataFrame
import pandas as pd
import re

import torch
from torch_geometric.data import Data, Dataset
from core.model.TNT import *

def get_dheading(goal_x, goal_y, cur_x, cur_y, cur_heading):
    if 0 < cur_heading < math.pi:  # Facing Left Half
        theta = cur_heading

    elif -(math.pi) < cur_heading < 0:  # Facing Right Half
        theta = 2 * math.pi + cur_heading

    elif cur_heading == 0:  # Facing up North
        theta = 0

    elif (cur_heading == math.pi) or (cur_heading == -(math.pi)):  # Facing South
        theta = 2 * math.pi + cur_heading

    trans_matrix = np.array(
        [[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]]
    )
    cur_pos = np.array([[cur_x], [cur_y]])
    goal_pos = np.array([[goal_x], [goal_y]])
    trans_cur = np.matmul(trans_matrix, cur_pos)
    trans_goal = np.matmul(trans_matrix, goal_pos)

    dx = trans_goal[0, 0] - trans_cur[0, 0]
    dy = trans_goal[1, 0] - trans_cur[1, 0]
    angle_diff = math.atan(abs(dy) / abs(dx))

    if dx > 0:
        if dy > 0:
            dheading = -(0.5 * math.pi - angle_diff)
        elif dy < 0:
            dheading = -(0.5*math.pi + angle_diff)
        else:
            dheading = -0.5*math.pi
    elif dx < 0:
        if dy > 0:
            dheading = (0.5 * math.pi - angle_diff)
        elif dy < 0:
            dheading = (0.5*math.pi + angle_diff)
        else:
            dheading = 0.5*math.pi
    else:
        if dy > 0:
            dheading = 0
        elif dy < 0:
            dheading = -math.pi
        else:
            dheading = 0

    return dheading

def s2a(single_vehicle_data, agent_obs):
    pass


def get_fc_edge_index(node_indices):
    """
    node_indices: np.array([indices]), the indices of nodes connecting with each other;
    return a tensor(2, edges), indicing edge_index
    """
    xx, yy = np.meshgrid(node_indices, node_indices)
    xy = np.vstack(([xx.reshape(-1), yy.reshape(-1)])).astype(np.int64)
    return xy


def get_traj_edge_index(node_indices):
    """
    generate the polyline graph for traj, each node are only directionally connected with the nodes in its future
    node_indices: np.array([indices]), the indices of nodes connecting with each other;
    return a tensor(2, edges), indicing edge_index
    """
    edge_index = np.empty((2, 0))
    for i in range(len(node_indices)):
        xx, yy = np.meshgrid(node_indices[i], node_indices[i:])
        edge_index = np.hstack([edge_index, np.vstack(([xx.reshape(-1), yy.reshape(-1)])).astype(np.int64)])
    return edge_index


class GraphData(Data):
    """
    override key `cluster` indicating which polyline_id is for the vector
    """

    def __inc__(self, key, value):
        if key == 'edge_index':
            return self.x.size(0)
        elif key == 'cluster':
            return int(self.cluster.max().item()) + 1
        else:
            return 0
    

# dataset loader which loads data into memory
class ArgoversetoGraph():
    def __init__(self, raw_data: DataFrame):
        self.raw_data = raw_data
        gc.collect()

    def process(self):
        """ transform the raw data and store in GraphData """
        # loading the raw data
        traj_lens = []
        valid_lens = []
        candidate_lens = []
        raw_data = self.raw_data

        # statistics
        traj_num = raw_data['feats'].values[0].shape[0]
        traj_lens.append(traj_num)

        lane_num = raw_data['graph'].values[0]['lane_idcs'].max() + 1
        valid_lens.append(traj_num + lane_num)

        candidate_num = raw_data['tar_candts'].values[0].shape[0]
        candidate_lens.append(candidate_num)
        
        num_valid_len_max = np.max(valid_lens)
        num_candidate_max = np.max(candidate_lens)

        # pad vectors to the largest polyline id and extend cluster, save the Data to disk
        # input data
        x, cluster, edge_index, identifier = self._get_x(raw_data)
        y = self._get_y(raw_data)
        graph_input = GraphData(
            x=torch.from_numpy(x).float(),
            y=torch.from_numpy(y).float(),
            cluster=torch.from_numpy(cluster).short(),
            edge_index=torch.from_numpy(edge_index).long(),
            identifier=torch.from_numpy(identifier).float(),    # the identify embedding of global graph completion

            traj_len=torch.tensor([traj_lens[0]]).int(),            # number of traj polyline
            valid_len=torch.tensor([valid_lens[0]]).int(),          # number of valid polyline
            time_step_len=torch.tensor([num_valid_len_max]).int(),    # the maximum of no. of polyline

            candidate_len_max=torch.tensor([num_candidate_max]).int(),
            candidate_mask=[],
            candidate=torch.from_numpy(raw_data['tar_candts'].values[0]).float(),
            candidate_gt=torch.from_numpy(raw_data['gt_candts'].values[0]).bool(),
            offset_gt=torch.from_numpy(raw_data['gt_tar_offset'].values[0]).float(),
            target_gt=torch.from_numpy(raw_data['gt_preds'].values[0][0][-1, :]).float(),

            orig=torch.from_numpy(raw_data['orig'].values[0]).float().unsqueeze(0),
            rot=torch.from_numpy(raw_data['rot'].values[0]).float().unsqueeze(0),
            seq_id=torch.tensor([int(raw_data['seq_id'])]).int(),
            num_graphs = 1,
            batch = None
        )
        return graph_input

    @staticmethod
    def _get_x(data_seq):
        """
        feat: [xs, ys, vec_x, vec_y, step(timestamp), traffic_control, turn, is_intersection, polyline_id];
        xs, ys: the control point of the vector, for trajectory, it's start point, for lane segment, it's the center point;
        vec_x, vec_y: the length of the vector in x, y coordinates;
        step: indicating the step of the trajectory, for the lane node, it's always 0;
        traffic_control: feature for lanes
        turn: twon binary indicator representing is the lane turning left or right;
        is_intersection: indicating whether the lane segment is in intersection;
        polyline_id: the polyline id of this node belonging to;
        """
        feats = np.empty((0, 10))
        edge_index = np.empty((2, 0), dtype=np.int64)
        identifier = np.empty((0, 2))

        # get traj features
        traj_feats = data_seq['feats'].values[0]
        traj_has_obss = data_seq['has_obss'].values[0]
        step = np.arange(0, traj_feats.shape[1]).reshape((-1, 1))
        traj_cnt = 0
        for _, [feat, has_obs] in enumerate(zip(traj_feats, traj_has_obss)):
            xy_s = feat[has_obs][:-1, :2]
            vec = feat[has_obs][1:, :2] - feat[has_obs][:-1, :2]
            traffic_ctrl = np.zeros((len(xy_s), 1))
            is_intersect = np.zeros((len(xy_s), 1))
            is_turn = np.zeros((len(xy_s), 2))
            polyline_id = np.ones((len(xy_s), 1)) * traj_cnt
            feats = np.vstack([feats, np.hstack([xy_s, vec, step[has_obs][:-1], traffic_ctrl, is_turn, is_intersect, polyline_id])])
            traj_cnt += 1

        # get lane features
        graph = data_seq['graph'].values[0]
        ctrs = graph['ctrs']
        vec = graph['feats']
        traffic_ctrl = graph['control'].reshape(-1, 1)
        is_turns = graph['turn']
        is_intersect = graph['intersect'].reshape(-1, 1)
        lane_idcs = graph['lane_idcs'].reshape(-1, 1) + traj_cnt
        steps = np.zeros((len(lane_idcs), 1))
        feats = np.vstack([feats, np.hstack([ctrs, vec, steps, traffic_ctrl, is_turns, is_intersect, lane_idcs])])

        # get the cluster and construct subgraph edge_index
        cluster = copy(feats[:, -1].astype(np.int64))
        for cluster_idc in np.unique(cluster):
            [indices] = np.where(cluster == cluster_idc)
            identifier = np.vstack([identifier, np.min(feats[indices, :2], axis=0)])
            if len(indices) <= 1:
                continue                # skip if only 1 node
            if cluster_idc < traj_cnt:
                edge_index = np.hstack([edge_index, get_fc_edge_index(indices)])
            else:
                edge_index = np.hstack([edge_index, get_fc_edge_index(indices)])
        return feats, cluster, edge_index, identifier

    @staticmethod
    def _get_y(data_seq):
        traj_obs = data_seq['feats'].values[0][0]
        traj_fut = data_seq['gt_preds'].values[0][0]
        offset_fut = np.vstack([traj_fut[0, :] - traj_obs[-1, :2], traj_fut[1:, :] - traj_fut[:-1, :]])
        return offset_fut.reshape(-1).astype(np.float32)

# function to convert the coordinates of trajectories from relative to world
def convert_coord(traj, orig, rot):
    traj_converted = np.matmul(np.linalg.inv(rot), traj.T).T + orig.reshape(-1, 2)
    return traj_converted
