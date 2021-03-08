import os
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')


def evaluation(gt_path, pred_path):
    """
    Args:
        gt_path (string) : root directory of ground truth file
        pred_path (string) : root directory of prediction file (output of inference.py)
    """
    results = {}
    for status in ['public', 'private']:
        gt = pd.read_csv(os.path.join(gt_path, f'{status}.csv'))
        pred = pd.read_csv(os.path.join(pred_path, f'{status}.csv'))

        cls_report = classification_report(gt.ans.values, pred.ans.values, labels=np.arange(18), output_dict=True)
        acc = cls_report['accuracy']
        f1 = np.mean([cls_report[str(i)]['f1-score'] for i in range(18)])

        results[status] = {'accuracy': acc, 'f1': f1}

    print(results)
    result = f'{results["private"]["accuracy"] * 100:.2f}%'

    return result


if __name__ == '__main__':
    gt_path = '/mnt/ssd/data/mask/mask_final/ground_truth'  # os.environ['SM_GROUND_TRUTH_DIR']
    pred_path = './outputs'  # os.environ['SM_OUTPUT_DATA_DIR']

    result = evaluation(gt_path, pred_path)
    print(f'Final Score: {result}')
