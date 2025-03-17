import pandas as pd
import numpy as np
from typing import Dict, List


def load_event_data(file_path: str):
    """
    Load and process event data from the CSV file.
    
    Returns:
        dict: A dictionary with event types as keys and lists of event times as values.
    """
    df = pd.read_csv(file_path)
    event_times = df.groupby('event_type')['event_time'].apply(list).to_dict()
    event_times = {key - 1: value for key, value in event_times.items()}
    return event_times


def generate_X_y_multitype(
        event_times_dict: Dict[int, List[float]], # A dictionary of event times for each event type.
        num_types: int,                           # The number of event types.
        start_periods: int,                       # The starting period for generating data.
        end_periods: int,                         # The ending period for generating data.
        k: int,                                   # The number of periods for prediction.
        period_length: int                        # The duration of each period.
        ):
    """
    Generate feature matrix X and labels y for multi-type event prediction.
    
    Returns:
        tuple: Two numpy arrays, one for features X and one for labels y.
    """
    X = []
    y = []
    
    for i in range(start_periods, end_periods - k):
        start_time = i * period_length
        end_time = (i + k) * period_length
        
        # Initialize an empty list to store events for each type
        events_in_period = [[] for _ in range(num_types)]
        
        for event_type in range(num_types):
            events_in_period[event_type] = [t for t in event_times_dict[event_type] if start_time <= t < end_time]
        
        # Construct features X
        X.append(events_in_period)

        next_period_start = (i + k) * period_length
        next_period_end = (i + k + 1) * period_length
        events_in_next_period = [[] for _ in range(num_types)]
        
        for event_type in range(num_types):
            events_in_next_period[event_type] = [t for t in event_times_dict[event_type] if next_period_start <= t < next_period_end]
        
        # Construct labels y
        event_count_per_type = [len(events_in_next_period[event_type]) for event_type in range(num_types)]
        first_event_time_per_type = [events_in_next_period[event_type][0] - next_period_start if len(events_in_next_period[event_type]) > 0 else 0 for event_type in range(num_types)]
        
        # Extract non-zero first event times
        non_zero_first_event_times = [time for time in first_event_time_per_type if time > 0]
        if non_zero_first_event_times:
            first_event_time = min(non_zero_first_event_times)
            first_event_type = first_event_time_per_type.index(first_event_time)
        else:
            first_event_time = 0
            first_event_type = -1
        
        y.append(event_count_per_type + first_event_time_per_type + [first_event_time, first_event_type])

    return np.array(X, dtype=object), np.array(y)


def pad_sequences(sequences, 
                  num_types: int,
                  max_len=None,       # The maximum length to pad sequences to. If None, the longest sequence length will be used.
                  padding_value=0):
    """
    Pad sequences to a consistent length with a specified padding value.
    
    Returns:
        np.ndarray: A padded numpy array of shape (n_samples, num_types, max_len).
    """
    if max_len is None:
        max_len = max(len(seq[j]) for seq in sequences for j in range(num_types))
    
    padded_sequences = np.full((len(sequences), num_types, max_len), padding_value, dtype=float)
    
    for i, seq in enumerate(sequences):
        for j in range(num_types):
            seq_len = len(seq[j])
            if seq_len > 0:
                padded_sequences[i, j, :seq_len] = seq[j]
    
    return padded_sequences