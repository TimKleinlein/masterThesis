import os
import pickle

def get_unique_abbreviation(name, existing_abbr):
    """Generate a unique abbreviation for the given name."""
    if len(name) < 2:
        return name  # Handle short names

    # Start with the first two letters
    abbreviation = name[:2]

    # If the abbreviation is already taken, try the first and third letters
    index = 3
    while abbreviation in existing_abbr:
        if len(name) >= index:
            abbreviation = name[0] + name[index - 1]
        else:
            abbreviation = name[:index]  # In case the name is shorter than the current index
        index += 1

    return abbreviation

def create_abbreviation_dict(directory):
    """Create a dictionary mapping each subdirectory name to a unique abbreviation."""
    abbr_dict = {}
    existing_abbr = set()

    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            abbr = get_unique_abbreviation(subdir, existing_abbr)
            abbr_dict[subdir] = abbr
            existing_abbr.add(abbr)

    return abbr_dict

def create_reverse_dict(abbr_dict):
    """Create a reverse dictionary mapping from abbreviations to full names."""
    reverse_dict = {abbr: name for name, abbr in abbr_dict.items()}
    return reverse_dict

# Usage example
directory_path = 'data/LibriSpeech/dev-clean'
abbr_dict = create_abbreviation_dict(directory_path)

# Save the dictionary as a pickle file
pkl_file_path = 'data/abbr_dict.pkl'
with open(pkl_file_path, 'wb') as pkl_file:
    pickle.dump(abbr_dict, pkl_file)

# Create and save the reverse dictionary
abbr_dict_reverse = create_reverse_dict(abbr_dict)
pkl_reverse_file_path = 'data/abbr_dict_reverse.pkl'
with open(pkl_reverse_file_path, 'wb') as pkl_reverse_file:
    pickle.dump(abbr_dict_reverse, pkl_reverse_file)

print(f"Abbreviation dictionary saved to {pkl_file_path}")
print(f"Reverse abbreviation dictionary saved to {pkl_reverse_file_path}")
