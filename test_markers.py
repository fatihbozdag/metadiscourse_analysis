# Test script to verify the fix for unhashable type error

# Define metadiscourse marker categories as tuples (hashable) instead of sets (unhashable)
INTERACTIVE_MARKERS = {
    'transitions': ('first', 'second', 'third'),
    'frame_markers': ('my purpose is', 'my aim is'),
    'endophoric_markers': ('noted above', 'see above')
}

INTERACTIONAL_MARKERS = {
    'hedges': ('might', 'perhaps', 'possible'),
    'boosters': ('certainly', 'definitely', 'clearly'),
    'attitude_markers': ('unfortunately', 'fortunately')
}

# Test if they are hashable
print("Testing if markers are hashable...")
test_dict = {}

for category in INTERACTIVE_MARKERS:
    print(f"Category: {category}, Type: {type(category)}")
    test_dict[category] = 1

for category in INTERACTIONAL_MARKERS:
    print(f"Category: {category}, Type: {type(category)}")
    test_dict[category] = 1

print("All markers are hashable!")
