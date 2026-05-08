from utils import package_results_for_submission_ek100

CFG_FILES = [
    ('expts/14_ek100_vjepa2_tsn_action_manual_best_test.txt', 0),
]
WTS = [1.0]
SLS = [1, 4, 4]

package_results_for_submission_ek100(CFG_FILES, WTS, SLS)