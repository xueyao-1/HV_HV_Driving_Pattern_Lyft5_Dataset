"""
Build a shareable trajectories + pattern-label + duration dataset for ALL
vehicles in the HV-HV (trainHH) dataset.

This reuses the exact rule-based segmentation/labelling algorithm from
`label_and_encode_vehicles_6000_8000.py` (same Segmentation class, same
thresholds and decision rules -- see that file's docstring for what was
changed vs. the original notebook and why). This script:

  1. Runs that labelling pass over every vehicle in trainHH_data.csv
     (not just a slice), producing the same VehicleID/Label/Point rows as
     before.
  2. From those Label/Point rows, derives a SEGMENTS table: one row per
     driving-pattern segment (VehicleID, segment_index, pattern_label,
     pattern_id, start_frame, end_frame, duration) -- this is exactly the
     (encoded_labels, durations) pairing used in the earlier encoding step,
     just laid out as a flat table instead of two parallel per-vehicle
     arrays.
  3. Emits a TRAJECTORIES table: one row per (vehicle, frame) with the raw
     smoothed trajectory variables the labelling algorithm reads
     (smooth_v_follow, a_follow, time_headway, smooth_delta_v), tagged with
     the segment_index it falls into -- so a reader can join trajectories
     and segments on (VehicleID, segment_index) to see the raw signal
     behind any labelled pattern.

FrameID is confirmed to be a 0-based, per-vehicle positional index (verified
against the source data), so segment boundaries (frame numbers from the
labelling step) can be used directly to bucket each frame of the raw
trajectory into its segment via a per-vehicle searchsorted -- no separate
alignment step needed.

Outputs (in OUTPUT_DIR):
  - labeled_data_all_vehicles.csv : VehicleID, Label, Point (all vehicles; same
                                     schema as the earlier range-based files)
  - segments_all_vehicles.csv     : VehicleID, segment_index, pattern_label,
                                     pattern_id, start_frame, end_frame, duration
  - trajectories_all_vehicles.csv : VehicleID, FrameID, smooth_v_follow,
                                     a_follow, time_headway, smooth_delta_v,
                                     segment_index
"""

import time
import traceback

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
SOURCE_CSV = "/Users/cielyao/surfdrive/Programming/Q2_Hetero_HDV/Dataprocessing/Lyft5_dataset/trainHH/trainHH_data.csv"
OUTPUT_DIR = "/Users/cielyao/surfdrive/Programming/Q3_DB_prediction/Data"

VEHICLE_SLICE = slice(None)  # all vehicles; set e.g. slice(0, 1000) to test on a subset

LABELS_CSV_OUT = f"{OUTPUT_DIR}/labeled_data_all_vehicles.csv"
SEGMENTS_CSV_OUT = f"{OUTPUT_DIR}/segments_all_vehicles.csv"
TRAJECTORIES_CSV_OUT = f"{OUTPUT_DIR}/trajectories_all_vehicles.csv"

NEEDED_COLUMNS = [
    "Vehicle_ID",
    "FrameID",
    "smooth_v_follow",
    "a_follow",
    "time_headway",
    "smooth_delta_v",
]

LABEL_ENCODING = {
    'Follow_behind': 0,
    'Slow_down': 1,
    'Catch_up': 2,
    'Speed_up': 3,
    'Fall_behind': 4,
    'Hold_speed': 5,
}

PROGRESS_EVERY = 500

# --------------------------------------------------------------------------
# Segmentation: identical rule-based logic to select_vehicle.ipynb /
# label_and_encode_vehicles_6000_8000.py (see that file for the efficiency
# notes -- generate_result_dataset() below only computes the requested
# vehicle via a pre-built groupby lookup instead of rescanning every
# vehicle in the dataset).
# --------------------------------------------------------------------------
class Segmentation():
    _vehicle_groups_cache = {}

    def __init__(self, all_data, desired_vehicle_id):
        self.all_data = all_data
        self.desired_vehicle_id = desired_vehicle_id

        cache_key = id(all_data)
        if cache_key not in Segmentation._vehicle_groups_cache:
            Segmentation._vehicle_groups_cache[cache_key] = {
                vid: sub_df for vid, sub_df in all_data.groupby('Vehicle_ID', sort=False)
            }
        self.vehicle_groups = Segmentation._vehicle_groups_cache[cache_key]

    def cal_inflection_points(self, data):
        local_minima = argrelextrema(data, np.less)
        local_maxima = argrelextrema(data, np.greater)
        inflection_points = np.sort(np.concatenate((local_minima[0], local_maxima[0])))
        inflection_indexs = [i for i in inflection_points]
        inflection_indexs.insert(0, 0)
        inflection_indexs.append(len(data) - 1)
        inflection_coordinates = [data[i] for i in inflection_indexs]
        delta_x = np.diff(inflection_indexs)
        delta_y = np.diff(inflection_coordinates)
        delta_x = np.append(delta_x, 0)
        delta_y = np.append(delta_y, 0)
        return inflection_indexs, inflection_coordinates, delta_x, delta_y

    def generate_result_dataset(self, variables):
        vehicle_id = self.desired_vehicle_id
        vehicle_data = self.vehicle_groups[vehicle_id]
        time_id = vehicle_data['FrameID'].values
        speed = vehicle_data[str(variables)].values
        inflection_points, inflection_coordinates, delta_x, delta_y = self.cal_inflection_points(speed)
        inflection_time_ids = [time_id[i] for i in inflection_points]
        result_dataset = pd.DataFrame({
            'VehicleID': vehicle_id,
            'InflectionTimeID': inflection_time_ids,
            'InflectionSpeed': inflection_coordinates,
            'DeltaX': delta_x,
            'DeltaY': delta_y
        })
        return result_dataset

    def find_data(self, result_dataset):
        single_vehicle_result = result_dataset[result_dataset['VehicleID'] == self.desired_vehicle_id]
        single_vehicle_data = self.vehicle_groups[self.desired_vehicle_id]
        delta_t = single_vehicle_result['DeltaX'].values
        delta_y = single_vehicle_result['DeltaY'].values
        time_turning_points = single_vehicle_result['InflectionTimeID'].values
        turning_points_values = single_vehicle_result['InflectionSpeed'].values
        return single_vehicle_data, delta_t, delta_y, time_turning_points, turning_points_values

    def get_states_set(self, delta_y, theta1, theta2):
        states = []
        for dy in delta_y:
            if dy >= theta1:
                state = 'I'
            elif dy <= theta2:
                state = 'D'
            else:
                state = 'K'
            states.append(state)
        return states

    def merge(self, states, delta_t, delta_y, time_turning_points, theta1, theta2, gama):
        merged_index_set = []
        for index, j in enumerate(states):
            if index < 1:
                if states[index + 1] != j:
                    if delta_t[index] <= gama and theta2 <= delta_y[index] <= theta1:
                        states[index] = states[index - 1]
                        merged_index_set.append(index)
            elif index == len(states) - 1:
                if states[index - 1] != j:
                    if delta_t[index] <= gama and theta2 <= delta_y[index] <= theta1:
                        states[index] = states[index - 1]
                        merged_index_set.append(index)
            else:
                if states[index - 1] != j and states[index + 1] != j:
                    if delta_t[index] <= gama and theta2 <= delta_y[index] <= theta1:
                        states[index] = states[index - 1]
                        merged_index_set.append(index)
                elif states[index - 1] == j and states[index + 1] != j:
                    if delta_t[index] <= gama and theta2 <= delta_y[index] <= theta1:
                        states[index] = states[index - 1]
                        merged_index_set.append(index)
                elif states[index + 1] == j and states[index - 1] != j:
                    if delta_t[index] <= gama and theta2 <= delta_y[index] <= theta1:
                        states[index] = states[index + 1]
                        merged_index_set.append(index)
        try:
            merged_index_set.remove(0)
        except Exception:
            pass
        try:
            merged_index_set.remove(len(time_turning_points) - 1)
        except Exception:
            pass
        time_turning_points_final = np.delete(time_turning_points, merged_index_set)
        return time_turning_points_final, merged_index_set

    def cal_states_final(self, states, variable, turning_point, merged_index_set):
        if str(variable) == 'a_follow':
            variable_threshold = 0.25
        elif str(variable) == 'smooth_v_follow':
            variable_threshold = 13
        elif str(variable) == 'time_headway':
            variable_threshold = 1.5
        elif str(variable) == 'smooth_delta_v':
            variable_threshold = 0.8
        else:
            variable_threshold = np.inf
        single_vehicle_data = self.vehicle_groups[self.desired_vehicle_id]
        for index, point in enumerate(turning_point):
            if states[index] == 'K':
                if index != turning_point.shape[0] - 1:
                    next_point = turning_point[index]
                else:
                    next_point = turning_point[-1]
                point -= turning_point[0]
                next_point -= turning_point[0]
                mean_variable = np.mean(single_vehicle_data[str(variable)][point:next_point + 1])
                if mean_variable >= variable_threshold or mean_variable <= -variable_threshold:
                    states[index] = 'K'
                else:
                    states[index] = 'K'
        states_final = [i for index, i in enumerate(states) if index not in merged_index_set]
        return states_final

    def rule_label(self, states_v_final, turning_points_final_v, states_a_final, turning_points_final_a, states_d_final, turning_points_final_d):
        rule_based_label = []
        rule_based_points = []
        for v_index, state_v in enumerate(states_v_final):
            if state_v == 'I':
                rule_based_label.append('Speed_up')
                rule_based_points.append(turning_points_final_v[v_index])
            elif state_v == 'D':
                rule_based_label.append('Slow_down')
                rule_based_points.append(turning_points_final_v[v_index])
            else:
                turning_point_v = turning_points_final_v[v_index]
                d_ = turning_points_final_d[turning_points_final_d <= turning_point_v].max() if np.any(turning_points_final_d <= turning_point_v) else None
                index_d_ = np.where(turning_points_final_d == d_)[0][0] if d_ is not None else None
                if turning_points_final_d[index_d_] in rule_based_points:
                    continue
                if len(rule_based_points) != 0:
                    if turning_points_final_d[index_d_] - rule_based_points[-1] < 10:
                        continue
                if states_d_final[index_d_] == 'I':
                    rule_based_label.append('Fall_behind')
                    rule_based_points.append(turning_points_final_d[index_d_])
                elif states_d_final[index_d_] == 'D':
                    rule_based_label.append('Catch_up')
                    rule_based_points.append(turning_points_final_d[index_d_])
                else:
                    turning_point_d = turning_points_final_d[index_d_]
                    a_ = turning_points_final_a[turning_points_final_a <= turning_point_d].max() if np.any(turning_points_final_a <= turning_point_d) else None
                    index_a_ = np.where(turning_points_final_a == a_)[0][0] if a_ is not None else None
                    if turning_points_final_a[index_a_] in rule_based_points:
                        continue
                    if len(rule_based_points) != 0:
                        if turning_points_final_a[index_a_] - rule_based_points[-1] < 10:
                            continue
                    if states_a_final[index_a_] == 'I':
                        rule_based_label.append('Follow_behind')
                        rule_based_points.append(turning_points_final_a[index_a_])
                    elif states_a_final[index_a_] == 'D':
                        rule_based_label.append('Follow_behind')
                        rule_based_points.append(turning_points_final_a[index_a_])
                    else:
                        rule_based_label.append('Hold_speed')
                        rule_based_points.append(turning_points_final_a[index_a_])
        return rule_based_label, rule_based_points


def label_one_vehicle(all_data, desired_vehicle_id):
    sg = Segmentation(all_data, desired_vehicle_id)

    result_dataset_velocity = sg.generate_result_dataset('smooth_v_follow')
    result_dataset_acceleration = sg.generate_result_dataset('a_follow')
    result_dataset_distance = sg.generate_result_dataset('time_headway')
    result_dataset_delta_v = sg.generate_result_dataset('smooth_delta_v')

    theta1_velocity = 1.5
    theta2_velocity = -theta1_velocity
    theta1_acceleration = 0.25
    theta2_acceleration = -theta1_acceleration
    theta1_distance = 0.15
    theta2_distance = -theta1_distance
    theta1_delta_v = 0.8
    theta2_delta_v = -theta1_delta_v
    gama = 20

    _, delta_t_v, delta_y_v, time_turning_points_v, _ = sg.find_data(result_dataset_velocity)
    _, delta_t_a, delta_y_a, time_turning_points_a, _ = sg.find_data(result_dataset_acceleration)
    _, delta_t_d, delta_y_d, time_turning_points_d, _ = sg.find_data(result_dataset_distance)
    _, delta_t_dv, delta_y_dv, time_turning_points_dv, _ = sg.find_data(result_dataset_delta_v)

    states_v = sg.get_states_set(delta_y_v, theta1_velocity, theta2_velocity)
    states_a = sg.get_states_set(delta_y_a, theta1_acceleration, theta2_acceleration)
    states_d = sg.get_states_set(delta_y_d, theta1_distance, theta2_distance)
    states_dv = sg.get_states_set(delta_y_dv, theta1_delta_v, theta2_delta_v)

    turning_points_final_v, merged_index_set_v = sg.merge(states_v, delta_t_v, delta_y_v, time_turning_points_v, theta1_velocity, theta2_velocity, gama)
    turning_points_final_a, merged_index_set_a = sg.merge(states_a, delta_t_a, delta_y_a, time_turning_points_a, theta1_acceleration, theta2_acceleration, gama)
    turning_points_final_d, merged_index_set_d = sg.merge(states_d, delta_t_d, delta_y_d, time_turning_points_d, theta1_distance, theta2_distance, gama)
    turning_points_final_dv, merged_index_set_dv = sg.merge(states_dv, delta_t_dv, delta_y_dv, time_turning_points_dv, theta1_delta_v, theta2_delta_v, gama)

    states_v_final = sg.cal_states_final(states_v, 'smooth_v_follow', time_turning_points_v, merged_index_set_v)
    states_a_final = sg.cal_states_final(states_a, 'a_follow', time_turning_points_a, merged_index_set_a)
    states_d_final = sg.cal_states_final(states_d, 'time_headway', time_turning_points_d, merged_index_set_d)

    rule_based_label, rule_based_points = sg.rule_label(
        states_v_final, turning_points_final_v,
        states_a_final, turning_points_final_a,
        states_d_final, turning_points_final_d,
    )
    return rule_based_label, rule_based_points, sg.vehicle_groups[desired_vehicle_id]


def run():
    print(f"Reading {SOURCE_CSV} ...")
    t0 = time.time()
    all_data_full = pd.read_csv(SOURCE_CSV, usecols=NEEDED_COLUMNS)
    print(f"  loaded {len(all_data_full):,} rows in {time.time() - t0:.1f}s")

    selected_vehicles = all_data_full['Vehicle_ID'].unique()
    target_vehicles = selected_vehicles[VEHICLE_SLICE]
    print(f"Target vehicles: {len(target_vehicles)} (of {len(selected_vehicles)} total)")

    all_data = all_data_full[all_data_full['Vehicle_ID'].isin(target_vehicles)].round(6)
    del all_data_full

    label_records = []
    segment_rows = []
    traj_frames = []
    failed = []

    t0 = time.time()
    for idx, vehicle_id in enumerate(target_vehicles, start=1):
        try:
            rule_based_label, rule_based_points, vehicle_frame_data = label_one_vehicle(all_data, vehicle_id)
            points = [int(p) for p in rule_based_points]
            label_records.append({'VehicleID': vehicle_id, 'Label': rule_based_label, 'Point': points})

            # Segment table: segment i spans [points[i], points[i+1]),
            # labelled by rule_based_label[i+1] (matches the established
            # encoding convention: labels[1:] paired with diff(points)).
            n_segments = len(points) - 1
            if n_segments > 0:
                for i in range(n_segments):
                    pattern_label = rule_based_label[i + 1]
                    segment_rows.append({
                        'VehicleID': vehicle_id,
                        'segment_index': i,
                        'pattern_label': pattern_label,
                        'pattern_id': LABEL_ENCODING.get(pattern_label, -1),
                        'start_frame': points[i],
                        'end_frame': points[i + 1],
                        'duration': points[i + 1] - points[i],
                    })

                # Trajectory rows tagged with the segment they fall into.
                # FrameID is a 0-based per-vehicle positional index, so a
                # plain searchsorted against the segment boundaries assigns
                # each frame to its segment.
                frame_ids = vehicle_frame_data['FrameID'].values
                boundaries = np.array(points[1:-1])  # interior boundaries only
                seg_idx = np.searchsorted(boundaries, frame_ids, side='right')
                seg_idx = np.clip(seg_idx, 0, n_segments - 1)
                traj = vehicle_frame_data[
                    ['Vehicle_ID', 'FrameID', 'smooth_v_follow', 'a_follow', 'time_headway', 'smooth_delta_v']
                ].copy()
                traj['segment_index'] = seg_idx
                traj_frames.append(traj)
        except Exception as e:
            failed.append(vehicle_id)
        if idx % PROGRESS_EVERY == 0 or idx == len(target_vehicles):
            elapsed = time.time() - t0
            print(f"  {idx}/{len(target_vehicles)} vehicles processed ({elapsed:.1f}s elapsed, {len(failed)} failed)")

    # --- Save labelled data (all vehicles) ---
    df_label = pd.DataFrame(label_records, columns=['VehicleID', 'Label', 'Point'])
    df_label.to_csv(LABELS_CSV_OUT, index=False)
    print(f"Saved {LABELS_CSV_OUT} ({len(df_label)} vehicles, {len(failed)} skipped)")

    # --- Save segments table ---
    df_segments = pd.DataFrame(segment_rows, columns=[
        'VehicleID', 'segment_index', 'pattern_label', 'pattern_id',
        'start_frame', 'end_frame', 'duration',
    ])
    df_segments.to_csv(SEGMENTS_CSV_OUT, index=False)
    print(f"Saved {SEGMENTS_CSV_OUT} ({len(df_segments)} segments)")

    # --- Save trajectories table ---
    df_traj = pd.concat(traj_frames, ignore_index=True)
    df_traj = df_traj.rename(columns={'Vehicle_ID': 'VehicleID'})
    df_traj.to_csv(TRAJECTORIES_CSV_OUT, index=False)
    print(f"Saved {TRAJECTORIES_CSV_OUT} ({len(df_traj)} frame rows)")

    if failed:
        print(f"Skipped {len(failed)} vehicle IDs (no usable segmentation): {failed[:50]}{' ...' if len(failed) > 50 else ''}")

    return df_label, df_segments, df_traj


if __name__ == '__main__':
    run()
