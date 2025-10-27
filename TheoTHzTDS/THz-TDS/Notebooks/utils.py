import os

def move_to_thz_tds_directory(max_levels=10):
    """
    Moves up directory levels until it finds the 'THz-TDS' folder, or handles errors if not found.
    
    Parameters:
    - max_levels: int, the maximum number of directory levels to move up (prevents infinite loop).
    
    Automatically prints errors or success messages to the user.
    """
    try:
        # Get the current working directory
        current_path = os.getcwd()

        # Keep track of the number of levels we've moved up
        levels_moved = 0

        # Loop until the path ends with 'THz-TDS' or we've moved up too many levels
        while not current_path.endswith('THz-TDS'):
            # If we've moved up more than max_levels, raise an error
            if levels_moved >= max_levels:
                raise FileNotFoundError(f"'THz-TDS' directory not found within {max_levels} levels of the current directory.")
            
            # Move up one directory
            parent_path = os.path.dirname(current_path)
            
            # If moving up doesn't change the path, we're at the root and 'THz-TDS' doesn't exist
            if parent_path == current_path:
                raise FileNotFoundError(f"'THz-TDS' directory not found in the current file tree.")
            
            current_path = parent_path
            levels_moved += 1

            # Change the working directory
            os.chdir(current_path)

        # Print confirmation of final directory
        print(f"Success! Current directory is now: {current_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
