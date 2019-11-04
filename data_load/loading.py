import pandas as pd

def get_convs(path):
    df = pd.read_csv(path)
    dialog_ids = df["dialogueID"].unique()
    starters = {}
    for i in range(len(df)):
        if not df["dialogueID"][i] in starters:
            starters[df["dialogueID"][i]] = df["from"][i]
    convs = [0] * len(dialog_ids)
    i = 0
    for dialog_id in dialog_ids:
        convs[i] = list(zip(df[df["dialogueID"] == dialog_id]["from"].tolist(),
                    df[df["dialogueID"] == dialog_id]["text"].tolist()))
        i += 1

    return convs
