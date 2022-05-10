import pandas as pd

columns = ["shape", "time (us)"]
def parse_file(file):
    with open(file) as f:
        content = f.read().splitlines()
    lines = []
    for i, line in enumerate(content):    
        if line.startswith("pt:"):
            shape = line.split(":")[1]
            dims = shape.split("_")
            kernel = dims[4]
            N = dims[5]
            iC = dims[6]
            H = dims[7]
            W = dims[8]
            oC = dims[9]
            groups = dims[10]
            
            dims_str = "Conv+ReLU: " + "kernel=" + kernel + ", N=" + N + ", iC=" + iC + ", H=" + H + ", W=" + W + ", oC=" + oC + ", groups=" + groups
            splited_element = [dims_str, line.split(":")[2].split(" ")[1]]
            lines.append(splited_element)
    df = pd.DataFrame(lines, columns=columns)
    df["time (us)"] = df["time (us)"].astype(float)
    print(df)
    return df

def compare(df1, df2):
    df = df1.merge(df2, on="shape", suffixes=["_fusion", "_no_fusion"])
    df["Gain"] = df["time (us)_no_fusion"] / df["time (us)_fusion"] - 1
    df["Gain"] = (df["Gain"] * 100).round(2).astype(str) + "%"
    print(df)
    

file_path_fusion = "with_fusion.log"
file_path_no_fusion = "no_fusion.log"

df1 = parse_file(file_path_fusion)
df2 = parse_file(file_path_no_fusion)

compare(df1, df2)
