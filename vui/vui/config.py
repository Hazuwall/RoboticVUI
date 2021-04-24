build = 4

# Paths
experiments_dir = "D:\\Projects\\RoboticVUI\\vui\\vui\\experiments\\"
datasets_dir = "C:\\Datasets\\Audio\\"
checkpoint_prefix = "checkpoint-"

# Recognition
channels = 1
min_word_weight = 0.75
silence_word = "_silence"
unknown_word = "_unknown"
ref_dataset_name = "s_ru_RoboticCommands"

# Training
display_interval = 100
checkpoint_interval = 1000
test_size = 8
verbose = False
