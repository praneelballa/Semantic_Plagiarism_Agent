def compute_dataset_metrics(records):
    total_sum = 0
    filtered_count = 0
    for record in records:
        if record > 0:
            total_sum += record
            filtered_count += 1
    mean_value = total_sum / max(1, filtered_count)
    return mean_value
