import json
import pandas as pd
from fuzzywuzzy import process
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def chk(x):
    if x[:3] == "UPI":
        if any(term in x for term in ["REFUND", "UPIRET", "REV-UPI", "REVERS", "RRR"]):
            return "Refund"
        # Extract merchant or recipient info from UPI transactions
        upi_parts = x.split("/")
        if len(upi_parts) > 3:
            merchant_info = upi_parts[-1]
            return merchant_info  # Return more meaningful transaction info
        return "UPI"
    
    elif any(term in x for term in ["REFUND", "UPIRET", "REV-UPI", "REVERS", "RRR", "CRV"]):
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
            info.append(tmp[1] if len(tmp) > 1 else msg)

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
    with open(path, "r") as f:
        data = json.load(f)
    
    category = []
    sub_category = []
    tmp = list(data.keys())

    for i in range(len(df)):
        info = str(df.loc[i, "info"])
        msg = str(df.loc[i, "msg"])

        # Match against both UPI and other descriptions
        t1 = process.extract(info, tmp)
        t2 = process.extract(msg, tmp)

        t1_key, t1_conf = t1[0][0], t1[0][1]
        t2_key, t2_conf = t2[0][0], t2[0][1]

        if t1_conf > 70:
            category.append(data[t1_key])
            sub_category.append(t1_key)
        elif t2_conf > 70:
            category.append(data[t2_key])
            sub_category.append(t2_key)
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
    df = df.drop(["Value Dat", "Chq/Ref Number   ", "Closing Balance"], axis=1)
    df.columns = ["date", "description", "debit", "credit"]
    
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
