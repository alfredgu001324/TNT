import math
import numpy as np
import pandas as pd
import os
import pickle
import argparse
import re
import copy
import matplotlib.pyplot as plt
import torch
from cubic_spline import Spline2D
from PIL import Image

torch.cuda.empty_cache()

OBS_LENGTH = 20
PRED_LENGTH = 30
FULL_LENGTH = OBS_LENGTH + PRED_LENGTH

def get_candidate_gt(target_candidate, gt_target):
    """
    find the target candidate closest to the gt and output the one-hot ground truth
    :param target_candidate, (N, 2) candidates
    :param gt_target, (1, 2) the coordinate of final target
    """
    displacement = gt_target - target_candidate
    gt_index = np.argmin(np.power(displacement[:, 0], 2) + np.power(displacement[:, 1], 2))

    onehot = np.zeros((target_candidate.shape[0], 1))
    onehot[gt_index] = 1

    offset_xy = gt_target - target_candidate[gt_index]
    return onehot, offset_xy

def get_ref_centerline(cline_list, pred_gt):
    if len(cline_list) == 1:
        return [Spline2D(x=cline_list[0][:, 0], y=cline_list[0][:, 1])], 0
    else:
        line_idx = 0
        ref_centerlines = [Spline2D(x=cline_list[i][:, 0], y=cline_list[i][:, 1]) for i in range(len(cline_list))]

        # search the closest point of the traj final position to each center line
        min_distances = []
        for line in ref_centerlines:
            xy = np.stack([line.x_fine, line.y_fine], axis=1)
            diff = xy - pred_gt[-1, :2]
            dis = np.hypot(diff[:, 0], diff[:, 1])
            min_distances.append(np.min(dis))
        line_idx = np.argmin(min_distances)
        return ref_centerlines, line_idx

def visualize_centerline(centerline) -> None:
    """Visualize the computed centerline.

    Args:
        centerline: Sequence of coordinates forming the centerline
    """
    line_coords = list(zip(*centerline))
    lineX = line_coords[0]
    lineY = line_coords[1]
    plt.plot(lineX, lineY, "--", color="grey", alpha=1, linewidth=1, zorder=0)
    plt.text(lineX[0], lineY[0], "s")
    plt.text(lineX[-1], lineY[-1], "e")
    plt.axis("equal")


def lane_candidate_sampling(centerline_list, orig, distance=0.5, viz=False):
    """the input are list of lines, each line containing"""
    candidates = []
    for lane_id, line in enumerate(centerline_list):
        sp = Spline2D(x=line[:, 0], y=line[:, 1])
        s_o, d_o = sp.calc_frenet_position(orig[0], orig[1])
        s = np.arange(s_o, sp.s[-1], distance)
        ix, iy = sp.calc_global_position_online(s)
        candidates.append(np.stack([ix, iy], axis=1))
    candidates = np.unique(np.concatenate(candidates), axis=0)

    if viz:
        fig = plt.figure(0, figsize=(8, 7))
        fig.clear()
        for centerline_coords in centerline_list:
            visualize_centerline(centerline_coords)
        plt.scatter(candidates[:, 0], candidates[:, 1], marker="*", c="g", alpha=1, s=6.0, zorder=15)
        plt.xlabel("Map X")
        plt.ylabel("Map Y")
        plt.axis("off")
        plt.title("No. of lane candidates = {}; No. of target candidates = {}; in green".format(len(centerline_list), len(candidates)))
        plt.savefig("asdf.png")
        plt.show()

    return candidates

def swap_left_and_right(
    condition: np.ndarray, left_centerline: np.ndarray, right_centerline: np.ndarray):
    """
    Swap points in left and right centerline according to condition.

    Args:
       condition: Numpy array of shape (N,) of type boolean. Where true, swap the values in the left and
                   right centerlines.
       left_centerline: The left centerline, whose points should be swapped with the right centerline.
       right_centerline: The right centerline.

    Returns:
       left_centerline
       right_centerline
    """

    right_swap_indices = right_centerline[condition]
    left_swap_indices = left_centerline[condition]

    left_centerline[condition] = right_swap_indices
    right_centerline[condition] = left_swap_indices
    return left_centerline, right_centerline


def centerline_to_polygon(
    centerline: np.ndarray, width_scaling_factor: float = 1.0, visualize: bool = False):
    """
    Convert a lane centerline polyline into a rough polygon of the lane's area.

    On average, a lane is 3.8 meters in width. Thus, we allow 1.9 m on each side.
    We use this as the length of the hypotenuse of a right triangle, and compute the
    other two legs to find the scaled x and y displacement.

    Args:
       centerline: Numpy array of shape (N,2).
       width_scaling_factor: Multiplier that scales 3.8 meters to get the lane width.
       visualize: Save a figure showing the the output polygon.

    Returns:
       polygon: Numpy array of shape (2N+1,2), with duplicate first and last vertices.
    """
    # eliminate duplicates
    _, inds = np.unique(centerline, axis=0, return_index=True)
    # does not return indices in sorted order
    inds = np.sort(inds)
    centerline = centerline[inds]

    dx = np.gradient(centerline[:, 0])
    dy = np.gradient(centerline[:, 1])

    # compute the normal at each point
    slopes = dy / dx
    inv_slopes = -1.0 / slopes

    thetas = np.arctan(inv_slopes)
    x_disp = 3.8 * width_scaling_factor / 2.0 * np.cos(thetas)
    y_disp = 3.8 * width_scaling_factor / 2.0 * np.sin(thetas)

    displacement = np.hstack([x_disp[:, np.newaxis], y_disp[:, np.newaxis]])
    right_centerline = centerline + displacement
    left_centerline = centerline - displacement

    # right centerline position depends on sign of dx and dy
    subtract_cond1 = np.logical_and(dx > 0, dy < 0)
    subtract_cond2 = np.logical_and(dx > 0, dy > 0)
    subtract_cond = np.logical_or(subtract_cond1, subtract_cond2)
    left_centerline, right_centerline = swap_left_and_right(subtract_cond, left_centerline, right_centerline)

    # right centerline also depended on if we added or subtracted y
    neg_disp_cond = displacement[:, 1] > 0
    left_centerline, right_centerline = swap_left_and_right(neg_disp_cond, left_centerline, right_centerline)

    if visualize:
        plt.scatter(centerline[:, 0], centerline[:, 1], 20, marker=".", color="b")
        plt.scatter(right_centerline[:, 0], right_centerline[:, 1], 20, marker=".", color="r")
        plt.scatter(left_centerline[:, 0], left_centerline[:, 1], 20, marker=".", color="g")
        # fname = datetime.datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S_%f")
        plt.savefig(f"ctr_polygon.png")
        plt.close("all")

    # return the polygon
    return convert_lane_boundaries_to_polygon(right_centerline, left_centerline)


def convert_lane_boundaries_to_polygon(right_lane_bounds, left_lane_bounds):
    """
    Take a left and right lane boundary and make a polygon of the lane segment, closing both ends of the segment.

    These polygons have the last vertex repeated (that is, first vertex == last vertex).

    Args:
       right_lane_bounds: Right lane boundary points. Shape is (N, 2).
       left_lane_bounds: Left lane boundary points.

    Returns:
       polygon: Numpy array of shape (2N+1,2)
    """
    assert right_lane_bounds.shape[0] == left_lane_bounds.shape[0]
    polygon = np.vstack([right_lane_bounds, left_lane_bounds[::-1]])
    polygon = np.vstack([polygon, right_lane_bounds[0]])
    return polygon


def transform_smarts_to_argoverse(dataset_path):
    
    obs = []
    actions = []
    regexp_agentid = re.compile(r".*_(.*).png")
    regexp_time = re.compile(r"(.*)_.*")
    c = 0
    total = 0
    # scenarios = os.listdir(dataset_path)
    scenarios = [scenario for scenario in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, scenario))]
    no_lanes = 0
    yes_lanes = 0
    seq_id = 0
    for scenario in scenarios:
        # print(c, scenario)
        # c += 1
        vehicles = set()
        scen_dir = os.path.join(dataset_path, scenario)
        for filename in os.listdir(scen_dir):
            if filename.endswith(".png"):
                match = regexp_agentid.search(filename)
                vehicles.add(match.group(1))
        # print(no_lanes, yes_lanes)
        for vehicle in vehicles:
            # total += 1
            with open(os.path.join(scen_dir, f"{vehicle}.pkl"), "rb") as f:
                vehicle_data = pickle.load(f)
            # asdf = [key for key in vehicle_data.keys()]
            
            # # import pdb; pdb.set_trace()
            # if len(vehicle_data[asdf[0]].road_waypoints.lanes) == 0:
            #     no_lanes += 1
            # else:
            #     import pdb; pdb.set_trace()
            #     yes_lanes += 1

            
            # print(vehicle_data[asdf[0]].road_waypoints)
            """
            sandbox
            """
            tnt_data = {}
            agt_traj = []
            agt_steps = []
            neighbourhood_traj = {}
            if len(vehicle_data) >= FULL_LENGTH:
                for t, time in enumerate(vehicle_data):
                    if t > FULL_LENGTH - 1: # argoverse dataset length
                        break
                    # AGENT's trajectory
                    agt_x = vehicle_data[time].ego_vehicle_state.position[0]
                    agt_y = vehicle_data[time].ego_vehicle_state.position[1]
                    agt_traj.append([agt_x, agt_y])
                    agt_steps.append(t)

                    # Neighbourhood vehicle's trajectory
                    for n_neighbour in range(len(vehicle_data[time].neighborhood_vehicle_states)):
                        other_v_id = vehicle_data[time].neighborhood_vehicle_states[n_neighbour].id
                        if other_v_id not in neighbourhood_traj:
                            neighbourhood_traj[other_v_id] = {'steps': [], 'traj': []}
                        other_v_x = vehicle_data[time].neighborhood_vehicle_states[n_neighbour].position[0]
                        other_v_y = vehicle_data[time].neighborhood_vehicle_states[n_neighbour].position[1]
                        neighbourhood_traj[other_v_id]['steps'].append(t)
                        neighbourhood_traj[other_v_id]['traj'].append([other_v_x, other_v_y])
            

                tnt_data['trajs'] = [np.array(agt_traj)]
                tnt_data['steps'] = [np.array(agt_steps)]

                timestamps = [key for key in vehicle_data.keys()]

                for other_v in neighbourhood_traj:
                    other_v_traj = np.array(neighbourhood_traj[other_v]['traj'])
                    tnt_data['trajs'].append(other_v_traj)
                    
                    other_v_steps = np.array(neighbourhood_traj[other_v]['steps'])
                    tnt_data['steps'].append(other_v_steps)
                
                orig = tnt_data['trajs'][0][OBS_LENGTH-1].copy().astype(np.float32) # get the 20's index (x,y)

                theta = float(vehicle_data[timestamps[OBS_LENGTH-1]].ego_vehicle_state.heading)
                rot = np.asarray([
                    [np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]], np.float32)

                "TODO agt_traj_obs, agt_traj_fut: DONE"
                agt_traj_obs = tnt_data['trajs'][0][0: OBS_LENGTH].copy().astype(np.float32)
                agt_traj_fut = tnt_data['trajs'][0][OBS_LENGTH:OBS_LENGTH+PRED_LENGTH].copy().astype(np.float32)
                agt_traj_fut = np.matmul(rot, (agt_traj_fut - orig.reshape(-1, 2)).T).T


                ctr_line_candts = []
                for i in range(len(vehicle_data[timestamps[0]].waypoint_paths)):
                    ctr_line_candt = []
                    for j in range(len(vehicle_data[timestamps[0]].waypoint_paths[i])):
                        x = vehicle_data[timestamps[0]].waypoint_paths[i][j].pos[0]
                        y = vehicle_data[timestamps[0]].waypoint_paths[i][j].pos[1]
                        ctr_line_candt.append([x, y])
                    ctr_line_candts.append(np.array(ctr_line_candt))
                
                for i, _ in enumerate(ctr_line_candts):
                    ctr_line_candts[i] = np.matmul(rot, (ctr_line_candts[i] - orig.reshape(-1, 2)).T).T
                
                tar_candts = lane_candidate_sampling(ctr_line_candts, [0, 0], viz=False)

                "TODO splines, ref_idx, tar_candts_gt, tar_offse_gt: DONE"
                splines, ref_idx = get_ref_centerline(ctr_line_candts, agt_traj_fut)
                tar_candts_gt, tar_offse_gt = get_candidate_gt(tar_candts, agt_traj_fut[-1])

                ""

                # import pdb; pdb.set_trace()

                feats, ctrs, has_obss, gt_preds, has_preds = [], [], [], [], []
                x_min, x_max, y_min, y_max = -1000, 1000, -1000, 1000
                for traj, step in zip(tnt_data['trajs'], tnt_data['steps']):
                    if OBS_LENGTH-1 not in step:
                        continue

                    # normalize and rotate
                    traj_nd = np.matmul(rot, (traj - orig.reshape(-1, 2)).T).T

                    # collect the future prediction ground truth
                    gt_pred = np.zeros((PRED_LENGTH, 2), np.float32) # shape of (PRED_LENGTH, 2) containing 0
                    has_pred = np.zeros(PRED_LENGTH, np.bool) # shape of (PRED_LENGTH,) containing False
                    future_mask = np.logical_and(step >= OBS_LENGTH, step < OBS_LENGTH + PRED_LENGTH) # shape of (50,) containing False upto OBS_LENGTH and True after
                    post_step = step[future_mask] - OBS_LENGTH 
                    post_traj = traj_nd[future_mask]
                    gt_pred[post_step] = post_traj # just containing np.array of shape (PRED_LENGTH, 2) with groud truth future traj
                    has_pred[post_step] = True

                    # colect the observation
                    obs_mask = step < OBS_LENGTH # shape of (50,) containing True upto OBS_LENGTH
                    step_obs = step[obs_mask]
                    traj_obs = traj_nd[obs_mask]
                    idcs = step_obs.argsort()
                    step_obs = step_obs[idcs]
                    traj_obs = traj_obs[idcs] # just containing observation traj

                    for i in range(len(step_obs)):
                        if step_obs[i] == OBS_LENGTH - len(step_obs) + i:
                            break
                    
                    step_obs = step_obs[i:]
                    traj_obs = traj_obs[i:]

                    if len(step_obs) <= 1:
                        continue

                    feat = np.zeros((OBS_LENGTH, 3), np.float32) # shape of (OBS_LENGTH, 3)
                    has_obs = np.zeros(OBS_LENGTH, np.bool) # shape of (OBS_LENGTH,)

                    feat[step_obs, :2] = traj_obs
                    feat[step_obs, 2] = 1.0 # feat first two columns are (x, y) the last one is just 1
                    has_obs[step_obs] = True
                    
                    if feat[-1, 0] < x_min or feat[-1, 0] > x_max or feat[-1, 1] < y_min or feat[-1, 1] > y_max: # -100 < x < 100, -100 < y < 100
                        continue

                    feats.append(feat)                  # displacement vectors
                    has_obss.append(has_obs)
                    gt_preds.append(gt_pred)
                    has_preds.append(has_pred)

                feats = np.asarray(feats, np.float32)
                has_obss = np.asarray(has_obss, np.bool)
                gt_preds = np.asarray(gt_preds, np.float32)
                has_preds = np.asarray(has_preds, np.bool)

                tnt_data['city'] = 'smarts_waymo'
                tnt_data['orig'] = orig
                tnt_data['theta'] = theta
                tnt_data['rot'] = rot
                tnt_data['feats'] = feats
                tnt_data['has_obss'] = has_obss

                tnt_data['has_preds'] = has_preds
                tnt_data['gt_preds'] = gt_preds
                tnt_data['tar_candts'] = tar_candts
                tnt_data['gt_candts'] = tar_candts_gt
                tnt_data['gt_tar_offset'] = tar_offse_gt

                tnt_data['ref_ctr_lines'] = splines         # the reference candidate centerlines Spline for prediction
                tnt_data['ref_cetr_idx'] = ref_idx          # the idx of the closest reference centerlines
                # import pdb; pdb.set_trace()
                

                """
                TODO data['graph']
                """
                # tnt_data['graph'] = get_lane_graph(tnt_data)


            
                ## data['graph'] part 
                x_min, x_max, y_min, y_max = -1000, 1000, -1000, 1000
                radius = max(abs(x_min), abs(x_max)) + max(abs(y_min), abs(y_max))
                lanes = {}
                timestamps = [key for key in vehicle_data.keys()]

                checked_lanes = []
                if len(vehicle_data[timestamps[0]].road_waypoints.lanes) != 0:
                    for time in timestamps:
                        asdf = [key for key in vehicle_data[time].road_waypoints.lanes]
                        if len(vehicle_data[time].road_waypoints.lanes[asdf[0]]) != 0:
                            # import pdb; pdb.set_trace()
                            for lane_id in vehicle_data[time].road_waypoints.lanes:
                                if lane_id in checked_lanes:
                                    continue
                                else:
                                    checked_lanes.append(lane_id)
                                    for j in range(len(vehicle_data[time].road_waypoints.lanes[lane_id])):
                                        lane_centerline = []
                                        for i in range(len(vehicle_data[time].road_waypoints.lanes[lane_id][j])):
                                            x = vehicle_data[time].road_waypoints.lanes[lane_id][j][i].pos[0]
                                            y = vehicle_data[time].road_waypoints.lanes[lane_id][j][i].pos[1]
                                            lane_centerline.append([x, y])
                                        lane_centerline = np.array(lane_centerline)
                                        centerline = np.matmul(tnt_data['rot'], (lane_centerline - tnt_data['orig'].reshape(-1, 2)).T).T
                                        x, y = centerline[:, 0], centerline[:, 1]
                                        if x.max() < x_min or x.min() > x_max or y.max() < y_min or y.min() > y_max:
                                            continue
                                        else:
                                            """Getting polygons requires original centerline"""
                                            polygon = centerline_to_polygon(lane_centerline[:, :2], visualize=True)
                                            # import pdb; pdb.set_trace()
                                            polygon = copy.deepcopy(polygon)
                                            lane_centerline = centerline
                                            lane_polygon = np.matmul(tnt_data['rot'], (polygon[:, :2] - tnt_data['orig'].reshape(-1, 2)).T).T
                                            if lane_id not in lanes:
                                                lanes[lane_id] = [lane_centerline, lane_polygon, "NONE", False, False] #[ctrln, polygon, turn_direction, has_traffic_contro, is_intersection]
                        else:
                            lane_id = 0
                            for i in range(len(vehicle_data[time].waypoint_paths)):
                                # import pdb; pdb.set_trace()
                                lane_centerline = []
                                for j in range(len(vehicle_data[time].waypoint_paths[i])):
                                    x = vehicle_data[time].waypoint_paths[i][j].pos[0]
                                    y = vehicle_data[time].waypoint_paths[i][j].pos[1]
                                    lane_centerline.append([x, y])
                                lane_centerline = np.array(lane_centerline)
                                centerline = np.matmul(tnt_data['rot'], (lane_centerline - tnt_data['orig'].reshape(-1, 2)).T).T
                                x, y = centerline[:, 0], centerline[:, 1]
                                if x.max() < x_min or x.min() > x_max or y.max() < y_min or y.min() > y_max:
                                        continue
                                else:
                                    """Getting polygons requires original centerline"""
                                    polygon = centerline_to_polygon(lane_centerline[:, :2], visualize=True)
                                    # import pdb; pdb.set_trace()
                                    polygon = copy.deepcopy(polygon)
                                    lane_centerline = centerline
                                    lane_polygon = np.matmul(tnt_data['rot'], (polygon[:, :2] - tnt_data['orig'].reshape(-1, 2)).T).T
                                    if lane_id not in lanes:
                                        lanes[lane_id] = [lane_centerline, lane_polygon, "NONE", False, False] #[ctrln, polygon, turn_direction, has_traffic_contro, is_intersection]
                                    lane_id += 1
                else:
                    lane_id = 0
                    for i in range(len(vehicle_data[timestamps[0]].waypoint_paths)):
                        # import pdb; pdb.set_trace()
                        lane_centerline = []
                        for j in range(len(vehicle_data[timestamps[0]].waypoint_paths[i])):
                            x = vehicle_data[timestamps[0]].waypoint_paths[i][j].pos[0]
                            y = vehicle_data[timestamps[0]].waypoint_paths[i][j].pos[1]
                            lane_centerline.append([x, y])
                        lane_centerline = np.array(lane_centerline)
                        centerline = np.matmul(tnt_data['rot'], (lane_centerline - tnt_data['orig'].reshape(-1, 2)).T).T
                        x, y = centerline[:, 0], centerline[:, 1]
                        if x.max() < x_min or x.min() > x_max or y.max() < y_min or y.min() > y_max:
                                continue
                        else:
                            """Getting polygons requires original centerline"""
                            polygon = centerline_to_polygon(lane_centerline[:, :2], visualize=True)
                            # import pdb; pdb.set_trace()
                            polygon = copy.deepcopy(polygon)
                            lane_centerline = centerline
                            lane_polygon = np.matmul(tnt_data['rot'], (polygon[:, :2] - tnt_data['orig'].reshape(-1, 2)).T).T
                            if lane_id not in lanes:
                                lanes[lane_id] = [lane_centerline, lane_polygon, "NONE", False, False] #[ctrln, polygon, turn_direction, has_traffic_contro, is_intersection]
                            lane_id += 1
                        

                lane_ids = list(lanes.keys())
                ctrs, feats, turn, control, intersect = [], [], [], [], []
                for lane_id in lane_ids:
                    lane = lanes[lane_id]
                    ctrln = lane[0]
                    num_segs = len(ctrln) - 1

                    ctrs.append(np.asarray((ctrln[:-1] + ctrln[1:]) / 2.0, np.float32))
                    feats.append(np.asarray(ctrln[1:] - ctrln[:-1], np.float32))

                    x = np.zeros((num_segs, 2), np.float32)
                    if lane[2] == 'LEFT':
                        x[:, 0] = 1
                    elif lane[2] == 'RIGHT':
                        x[:, 1] = 1
                    else:
                        pass
                    turn.append(x)
                    control.append(lane[3] * np.ones(num_segs, np.float32))
                    intersect.append(lane[4] * np.ones(num_segs, np.float32))
                lane_idcs = []
                count = 0
                for i, ctr in enumerate(ctrs):
                    lane_idcs.append(i * np.ones(len(ctr), np.int64))
                    count += len(ctr)
                num_nodes = count
                try:
                    lane_idcs = np.concatenate(lane_idcs, 0)
                except:
                    # import pdb; pdb.set_trace()
                    pass

                graph = {}
                graph['ctrs'] = np.concatenate(ctrs, 0)
                graph['num_nodes'] = num_nodes
                graph['feats'] = np.concatenate(feats, 0)
                graph['turn'] = np.concatenate(turn, 0)
                graph['control'] = np.concatenate(control, 0)
                graph['intersect'] = np.concatenate(intersect, 0)
                graph['lane_idcs'] = lane_idcs
                
                tnt_data['graph'] = graph

                # tnt_data['seq_id'] = scenario + vehicle
                tnt_data['seq_id'] = seq_id
                seq_id += 1

                df = pd.DataFrame(
                    [[tnt_data[key] for key in tnt_data.keys()]],
                    columns=[key for key in tnt_data.keys()]
                )
                # import pdb; pdb.set_trace()
                fname = f"features_{seq_id}.pkl"
                df.to_pickle(os.path.join('/root_data/sandbox_s2a/interm_data/train_intermediate/raw', fname))
                print("Saved {}".format(fname))
                """
                """
    # import pdb; pdb.set_trace()
    pass
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        required=True,
        type=str,
    )

    args = parser.parse_args()

    transform_smarts_to_argoverse(args.dataset_path)

    # python smarts_to_argoverse.py --dataset_path /home/data/hidden_dataset/validation_offline_dataset