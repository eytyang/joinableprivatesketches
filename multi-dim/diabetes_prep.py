import pandas as pd

from experiment import Experiment
from experiment_utils import center, prep_data, uniform_subsample, plot_results

import warnings
warnings.filterwarnings("ignore", message = "A column-vector y was passed when a 1d array was expected.")
warnings.filterwarnings("ignore", message = "X has feature names")

if __name__ == "__main__":
	# Save this file in case we need to do some hallucination modeling with the other diabetes datasets. 
