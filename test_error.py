# Test script to identify the unhashable type error

import sys

try:
    # Try to import the problematic part
    from analyze_metalanguage import INTERACTIVE_MARKERS, INTERACTIONAL_MARKERS
    print("Successfully imported marker categories")
    print(f"Type of INTERACTIVE_MARKERS: {type(INTERACTIVE_MARKERS)}")
    print(f"Type of INTERACTIONAL_MARKERS: {type(INTERACTIONAL_MARKERS)}")
    
    # Test if they are hashable
    test_dict = {}
    for category in INTERACTIVE_MARKERS:
        test_dict[category] = 1
    print("INTERACTIVE_MARKERS are hashable")
    
    for category in INTERACTIONAL_MARKERS:
        test_dict[category] = 1
    print("INTERACTIONAL_MARKERS are hashable")
    
except Exception as e:
    print(f"Error: {e}")
    print(f"Error type: {type(e)}")
    print(f"Error occurred at line: {sys.exc_info()[2].tb_lineno}")
