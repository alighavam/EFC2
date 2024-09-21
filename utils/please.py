'''
Bunch of helper functions to make my life easier. It's always 
nice to say please when asking for help.
'''

import os

def list_to_int(num_list: list[int]) -> int:
    """
    Convert a list of numbers into a single integer.
    
    Args:
    List of numbers to be concatenated.
    
    Returns:
    The concatenated integer.
    """
    # Convert each number to a string and concatenate them
    concatenated_str = ''.join(map(str, num_list))
    
    # Convert the concatenated string back to an integer
    return int(concatenated_str)