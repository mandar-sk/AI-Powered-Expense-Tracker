import json
import pandas as pd
from fuzzywuzzy import process
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def chk(x):
    if x[:3] == "UPI":
        if (
            ("REFUND" in x)
            or ("UPIRET" in x)
            or ("REV-UPI" == x[:7])
            or ("REVERS" in x)
            or ("RRR" in x)
        ):
            return "Refund"
        return "UPI"
    elif (
        ("REFUND" in x)
        or ("UPIRET" in x)
        or ("REV-UPI" == x[:7])
        or ("REVERS" in x)
        or ("RRR" in x)
        or ("CRV" in x)
    ):
        return "Refund"
    elif x[:3] == "ATW":
        return "ATM"
    elif x[:3] == "POS":
        return "Card"
    elif "INTEREST" in x:
        return "Savings Interest"
    elif x[:4] == "PRIN":
        return "FD returns"
    elif x[:4] == "INT.":
        return "FD interest"
    elif "AUTO SWEEP" in x:
        return "FD"
    elif (x[0].isdigit()) or (x[:4] == "IMPS") or (x[:4] == "NEFT") or ("FT " in x):
        return "Account Transfer"
    else:
        return "Others"


def get_descriptions(df):
    user_info = []
    info = []

    for i in range(len(df)):
        desc = str(df.loc[i, "description"]).strip() if pd.notna(df.loc[i, "description"]) else ""
        txn_type = df.loc[i, "type"]

        if txn_type == "UPI":
            tmp = desc.split("-")
            msg = tmp[-1] if len(tmp) >= 1 else "UPI"
            if msg[:3].upper() == "UPI":
                msg = "UPI"
            user_info.append(msg)
            info.append(tmp[1] if len(tmp) > 1 else msg)  # SAFE access

        elif txn_type == "Card":
            tmp = desc.split(" ")
            tmp = tmp[2:] if len(tmp) > 2 else []
            msg = " ".join(tmp).strip()
            user_info.append("Card")
            info.append(msg if msg else "Card")

        elif txn_type == "Refund":
            if df.loc[i, "credit"] == 0.0:
                df.loc[i, "type"] = "Others"
            user_info.append("Others")
            info.append("Others")

        else:
            user_info.append("Others")
            info.append(txn_type)

    df["info"] = info
    df["msg"] = user_info
    return df



def get_category(df, path):
    f = open(path, "r")
    data = json.load(f)
    tmp = list(data.keys())
    category = []
    sub_category = []

    for i in range(len(df)):
        info = str(df.loc[i, "info"])
        msg = str(df.loc[i, "msg"])

        if msg == "UPI" or msg == "Card" or msg == "Others":
            t1 = process.extract(info, tmp)
            t_key, t_conf, t_data = t1[0][0], t1[0][1], info
        else:
            t1 = process.extract(info, tmp)
            t2 = process.extract(msg, tmp)
            t1_key, t1_conf = t1[0][0], t1[0][1]
            t2_key, t2_conf = t2[0][0], t2[0][1]
            t_key, t_conf, t_data = (
                [t1_key, t1_conf, info] if t1_conf > t2_conf else [t2_key, t2_conf, msg]
            )

        if t_conf > 70:
            category.append(data[t_key])
            sub_category.append(t_key)
        else:
            if msg == "UPI" or msg == "Card":
                category.append(msg + " Transfer")
                sub_category.append(msg + " Transfer")
            else:
                category.append("Others")
                sub_category.append("Others")

    df["sub_category"] = sub_category
    df["category"] = category
    return df


def plot(x, y, type, filepath):
    if len(x) == 0 or len(y) == 0:
        print("No data found for ", type)
        return

    percent = 100.0 * y / y.sum()
    cdict = dict(zip(x, plt.cm.tab10.colors if len(x) <= 10 else plt.cm.tab20.colors))

    patches, texts = plt.pie(
        y,
        startangle=90,
        radius=50,
        colors=None if len(x) > 20 else [cdict[v] for v in x],
    )

    labels = ["{0} - {1:.2f}".format(i, j) for i, j in zip(x, y)]

    sort_legend = True
    if sort_legend:
        patches, labels, dummy = zip(
            *sorted(zip(patches, labels, y), key=lambda x: x[2], reverse=True)
        )

    plt.axis("equal")
    plt.title(type + " Plot")
    plt.legend(patches, labels, loc="upper left", bbox_to_anchor=(-1, 1.0), fontsize=11)

    # plt.savefig(filepath+type+".png", bbox_inches='tight')
    # plt.show()
def parse_date_flexibly(date_str):
    for fmt in ("%d/%m/%y", "%d-%m-%Y", "%Y-%m-%d"):
        try:
            return pd.to_datetime(date_str, format=fmt)
        except:
            continue
    return pd.NaT  # If all formats fail


def preprocess(df, month_idx):
    # Define expected mappings based on actual CSV format
    rename_mapping = {
        "Date": "date",
        "narration": "description",
        "Debit Amount": "debit",
        "Credit Amount": "credit"
    }

    # Rename columns if they exist in the dataset
    df = df.rename(columns=rename_mapping)

    # Drop unnecessary columns if they exist
    drop_cols = ["Value Date", "Chq/Ref Number", "Closing Balance"]
    df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors="ignore")

    # Ensure only expected columns exist
    required_cols = ["date", "description", "debit", "credit"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns! Found: {df.columns.tolist()}")

    # Clean and parse dates
    df["date"] = df["date"].astype(str).str.strip()
    df["date"] = df["date"].apply(parse_date_flexibly)

    # Handle missing values in debit & credit columns
    df["debit"] = df["debit"].fillna(0)
    df["credit"] = df["credit"].fillna(0)

    df["month"] = df["date"].dt.strftime("%B %Y")
    if month_idx != 0:
        df = df[df["date"].dt.month == month_idx]
    df = df.reset_index(drop=True)

    df["type"] = df["description"].apply(chk)
    df = get_descriptions(df)
    df = get_category(df, "data/data.json")

    return df




def get_dataframes(df, month_idx):
    df = preprocess(df, month_idx)
    df_debit = df[df.credit == 0.0]
    df_debit.reset_index(inplace=True, drop=True)
    df_debit.drop(columns=["credit"], inplace=True)

    df_credit = df[df.debit == 0.0]
    df_credit.reset_index(inplace=True, drop=True)
    df_credit.drop(columns=["debit"], inplace=True)
    if len(df) == 0:
        return df, df_debit, df_credit, pd.DataFrame()
    df_sum = df.sum(axis=0, numeric_only=True).to_frame(name="Sum")
    df_sum.index = ["Debit", "Credit"]
    df_sum.reset_index(inplace=True)

    return df, df_debit, df_credit, df_sum
