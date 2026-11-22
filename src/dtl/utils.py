
# implementation based on https://github.com/febo/P45/blob/main/p45.py

import argparse
from datetime import datetime
import math
import numpy as np
import pandas as pd
from scipy.stats import norm
import random

from collections import Counter
from json import load
from pandas.api.types import is_numeric_dtype

# KONSTANTA
# ukuran minimum antar nilai
DELTA = 1e-5
# minumum number of instances for each node
MINIMUM_INSTANCES = 2
# limit for the minimum number of instances in a continuous interval
INTERVAL_LIMIT = 25
# precision correction
CORRECTION = 1e-10
# batas value untuk threshold dan perhitungan error
EPSILON = 1e-3
# konstanta nilai 
ZERO = 1e-7

def _validate_weights(
    data :pd.DataFrame, weights: np.ndarray | None
) -> np.ndarray :
    if weights is None:
        return np.ones(len(data), dtype=float)

    weights_array = np.asanyarray(weights, dtype=float)
    
    if len(weights_array) != len(data):
        raise ValueError (
            f"Panjang weights ({len(weights_array)}) harus sama dengan "
            f"jumlah baris data ({len(data)})."
        )
    return weights_array


# Gain ratio

def gain_ratio(data : pd.DataFrame, attribute : str, weights : np.array | None = None) -> tuple[float, float, float]:
    """Menghitung gain ratio dari sebuah data

    Args:
        data (pd.DataFrame): data yang di prosess
        attribute (str): nama attribut yang di prosess
        weights (np.array | None, optional): bobot per atribut. Defaults to None.

    Returns:
        tuple[float, float, float]: berupa gain_ratio, gain, split_information
    """
    
    # inisialisasi weight dengan 1 jika tidak ada parameter
    weights = _validate_weights(weights, )
    
    # nilai kelas yang ada di Data
    target_column = data.iloc[:, - 1]
    is_valid_mask = data[attribute].notna()
    is_missing_mask = data[attribute].isna()
    
    total_weight = weights.sum()
    missing_weight = weights[is_missing_mask].sum()
    
    valid_weight = total_weight - missing_weight
    
    unique_classes = target_column[is_valid_mask].unique()
    
    # menghitung total Entropy
    total_class_entropy = 0.0
    for target_class in unique_classes:
        class_mask = (target_column == target_class) & is_valid_mask
        class_weight = weights[class_mask].sum()
        
        if class_weight > 0:
            class_probability = class_weight / valid_weight
            total_class_entropy -= class_probability * np.log2(class_probability)
        
    # Menghitung Partition Entropy & Split Information
    unique_attribute_values = data.iloc(is_valid_mask, attribute).unique()
    weighted_partition_entropy = 0
    split_information = 0.0
    
    for attr_value in unique_attribute_values:
        partition_mask = data[attribute] == attr_value
        partition_weight = weights[partition_mask].sum()
        partition_entropy = 0.0
        
        for target_class in unique_classes:
            class_in_partition_mask   = (target_class == target_class) & partition_mask
            class_in_partition_weight = weights[class_in_partition_mask].sum()
            
            if class_in_partition_weight > 0:
                probability = class_in_partition_weight/partition_weight
                partition_entropy -= probability * np.log2(probability)

        # Akumulasi Entropy Gain
        partition_ratio_valid = partition_weight / valid_weight
        weighted_partition_entropy += partition_ratio_valid * partition_entropy
        
        # Akumulasi Split Info
        partition_ratio_total = partition_weight / total_weight
        split_information -= partition_ratio_total * np.log2(partition_ratio_total)
    
    # pengurangan terhadap informasi hilang
    if missing_weight > 0:
        missing_ratio = missing_weight / total_weight
        split_information -= missing_ratio * np.log2(missing_ratio)
    
    valid_data_ratio = valid_weight / total_weight
    information_gain = valid_data_ratio * (total_class_entropy - weighted_partition_entropy)
    
    gain_ratio = (0.0 if (information_gain == 0 or split_information == 0) else information_gain/split_information)
    return gain_ratio, information_gain, split_information
            
def find_candidate_thresholds(data : pd.DataFrame, attribute : str,weights : np.array | None = None):
    """
        Generasi kandidate nilai untuk attribut data kontigu
    """
    weights = _validate_weights(weights, data)
    is_valid_mask = data[attribute].notna()
    valid_weights = weights[is_valid_mask]
    sorted_value_weight_pairs = sorted(
        zip(data.loc[is_valid_mask, attribute].to_numpy(), valid_weights), key=lambda pair:pair[0]
    )
    
    total_valid_weight = valid_weights.sum()
    unique_classes_count = data.iloc[:, -1][is_valid_mask].nunique()
    
    # ukuran minimum instance per interval berdasarkan C4.5
    raw_min_split = 0.1 * (total_valid_weight / unique_classes_count)
    min_instances_per_side = np.clip(
        raw_min_split, MINIMUM_INSTANCES, INTERVAL_LIMIT
    )
    
    candidate_thresholds : list[float] = []
    left_side_weight = 0.0
    right_side_weight = total_valid_weight
    
    for i in range(len(sorted_value_weight_pairs) - 1):
        current_value, current_weight = sorted_value_weight_pairs[i]
        next_value, _ = sorted_value_weight_pairs[i + 1]
        
        left_side_weight  += current_weight
        right_side_weight -= current_weight
        
        is_distinct_value = (current_value + DELTA < next_value)
        has_enough_left = (left_side_weight + CORRECTION) >= min_instances_per_side
        has_enough_right = (right_side_weight + CORRECTION) >= min_instances_per_side
        
        if is_distinct_value and has_enough_left and has_enough_right:
            midpoint = (current_value + next_value) / 2.0
            candidate_thresholds.append(midpoint)
    
    return candidate_thresholds

def gain_ratio_numeric(data : pd.DataFrame, attribute:str, weights:np.array) -> tuple[float, float, float]:
    """Melakukan evaluasi banyak threshold untuk secara dinamik melakukan diskritisasi 
       nilai kontigu pada atribut dan mengkalkulasikan informasi gain

    Args:
        data (pd.DataFrame): Data utama
        attribute (str): Nama Atribut
        weights (np.array): Nilai weights yang digunakan dalam kalkulasi

    Returns:
        tuple[float, float, float]: dari sebuah attribut
    """
    weights = _validate_weights(data, weights)
    
    thresholds = find_candidate_thresholds(data, attribute, weights)
    
    if len(thresholds) == 0:
        return 0,0,0
    
    valid_data = data[data[attribute].notna()].copy().reset_index(drop=True)
    valid_weights = weights[data[attribute].notna()]
    
    values = valid_data[attribute].copy()
    
    known_weights = valid_weights.sum()
    total_weights = weights.sum()
    
    gain_correction = known_weights/total_weights
    penalty = np.log2(len(thresholds))/weights.sum()
    gain_information = []
    
    for t in thresholds:
        binary_split = ['H' if v > t else 'L' for v in values]
        valid_data[attribute] = binary_split
        _, gain, split = gain_ratio(valid_data, attribute, valid_weights)
        
        # melakukan penalty saat melakukan evaluasi banyak threshold berdasarkan C4.5
        gain = (gain_correction * gain) - penalty
        ratio = 0 if gain == 0 or split == 0 else gain / split
        
        gain_information.append((ratio, gain))
    
    thresholds_gain = [g[1] for g in gain_information]
    selected = np.argmax(thresholds_gain)
    
    
    return gain_information[selected][0], gain_information[selected][1],thresholds_gain[selected]

def search_best_attribute(data : pd.DataFrame, weights : np.array) -> tuple[str, tuple[float, float, float]]:
    
    weights = _validate_weights(data, weights)
    
    predictors = data.iloc[:,0:- 1]
    if len(predictors.columns) == 0:
        return None, (0,0,0)
    
    candidate = []
    average_gain = 0
    
    for attribute in predictors.columns:
        if is_numeric_dtype(data[attribute]):
            c = attribute, gain_ratio_numeric(data,attribute, weights)
        else:
            c = attribute, gain_ratio(data,attribute, weights)

        # Prosess gain
        if c[1][1] > 0:
            average_gain += c[1][1]
            candidate.append(c)
        
    
    if len(candidate) == 0:
        return None, (0,0,0)

    average_gain = (average_gain / len(candidate)) - EPSILON
    selected = None
    
    
    best = (ZERO, ZERO, ZERO)
    
    for attribute, c in candidate:
        if c[0] > best[0] and c[1] >= average_gain:
            selected = attribute
            best = c
    
    return selected, best


def compute_z_squared(confidence_factor: float= 0.25) -> float:
    """Menghitung kuadrat z-score dari confidence factor untuk Pessimistic Error Pruning."""
    if not (0.0 < confidence_factor < 1.0):
        raise ValueError("confidence_factor harus berada di antara 0 dan 1.")
    return float(norm.ppf(1.0 - confidence_factor)**2)

def estimate_error(total : float, errors: float, confidence_factor :float = 0.25) -> float:
    """Menghitung estimasi penalti error pesimistis (C4.5 Pessimistic Error Pruning).

    Args:
        total (float): Total bobot/jumlah data yang masuk ke node.
        errors (float): Jumlah bobot/data yang salah diprediksi pada node.
        confidence_factor (float, optional): Tingkat kepercayaan statistik (CF) untuk pemangkasan.. Defaults to 0.25.

    Returns:
        float: Nilai penalti tambahan di atas error aktual.
    """
    
    if total == 0:
        return 0.0
    
    upper_limit = (1.0 - math.exp(math.log(confidence_factor) / total))
    
    # Tidak ada erorr sama sekali
    if errors < DELTA:
        return total * upper_limit
    
    # Error pecahan akibat distribusi bobot missing value
    if errors < 1 - DELTA:
        zero_error_penalty = total * upper_limit
        one_error_penalty = estimate_error(total , 1, confidence_factor)
        return zero_error_penalty + errors * (one_error_penalty - zero_error_penalty)

    if errors + 0.5 >= total:
        return 0.67 * (total - errors)
    
    z_sq = compute_z_squared(confidence_factor)
    adjusted_errors = errors * 0.5
    
    numerator = (
        adjusted_errors
        + (z_sq / 2.0)
        + math.sqrt(
            z_sq
            * (
                adjusted_errors * (1.0 - adjusted_errors / total)
                + (z_sq / 4.0)
            )
        )
    )
    upper_error_rate = numerator / (total + z_sq)

    return (total * upper_error_rate) - errors
        
    

