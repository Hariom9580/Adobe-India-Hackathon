import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report

def evaluate(pred_dir, gt_dir):
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    all_y_true = []
    all_y_pred = []
    for pred_file in pred_dir.glob("*_labeled.csv"):
        base = pred_file.stem.replace("_labeled", "")
        gt_file = gt_dir / f"{base}_labeled.csv"
        if not gt_file.exists():
            print(f"Missing ground truth for {base}")
            continue
        pred_df = pd.read_csv(pred_file)
        gt_df = pd.read_csv(gt_file)
        # Align by text and page (or by row if order is guaranteed)
        for _, pred_row in pred_df.iterrows():
            matches = gt_df[(gt_df['text'] == pred_row['text']) & (gt_df['page'] == pred_row['page'])]
            if not matches.empty:
                gt_label = matches.iloc[0]['label']
                all_y_true.append(gt_label)
                all_y_pred.append(pred_row['label'])
    print(classification_report(all_y_true, all_y_pred, zero_division=0))

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        evaluate(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python evaluate_metrics.py <predicted_labeled_dir> <ground_truth_labeled_dir>") 