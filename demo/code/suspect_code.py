def compute_mean(values):
    # Cosmetic variable renaming and refactored identifier names
    accumulator = 0
    for val in values:
        accumulator += val
    return accumulator / len(values)
