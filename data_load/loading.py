import pandas as pd

def get_convs(path):
    df = pd.read_csv(path)
    dialog_ids = df["dialogueID"].unique()
    convs = []
    for dialog_id in dialog_ids:
        convs.append(df[df["dialogueID"] == dialog_id]["text"].tolist())
    return convs
