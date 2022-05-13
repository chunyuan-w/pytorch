import argparse
import json

import pandas as pd

pd.options.display.max_colwidth = 80

BATCH_SIZE = 1

dict_type = {
    "onednn_verbose": {
        "name": "onednn_verbose",
        "len": 11,
        "header": ["info", "prim_template:operation", "engine", "primitive", "implementation", "prop_kind", "memory_descriptors", "attributes", "auxiliary", "problem_desc", "exec_time"]
    },
}


def preprocess(file, verbose_type="onednn_verbose"):
    with open(file) as f:
        content = f.read().splitlines()
    reorders = []
    for i, line in enumerate(content):
        if line.startswith(dict_type[verbose_type]["name"]) and len(line.split(',')) == dict_type[verbose_type]["len"]:
            reorder = line.split(",")
            reorders.append(reorder)

    df = pd.DataFrame(reorders, columns=dict_type[verbose_type]["header"])
    return df
    

def parse_line(line):
    
    parsed = line.split("_")
    # mb = parsed[0]
    mb = BATCH_SIZE
    channel = parsed[1]
    height = parsed[2]
    width = parsed[3]
    
    ic = channel.split("ic")[1].split("oc")[0]
    oc = channel.split("oc")[1]
    
    ih = height.split("ih")[1].split("oh")[0]
    kh = height.split("kh")[1].split("sh")[0]
    sh = height.split("sh")[1].split("dh")[0]
    # PT dilates = oneDNN dilates + 1
    dh = height.split("dh")[1].split("ph")[0]
    
    ph = height.split("ph")[1]
    
    print("#" * 50)
    print(sh)
    print(dh)
    print(ph)
    # kernel_size, N, iC, H, W, oC, groups
    # return [int(kh), mb, int(ic), int(ih), int(ih), int(oc)]
    # kernel_size, N, iC, H, W, oC, padding, stride, dilation
    return [int(kh), mb, int(ic), int(ih), int(ih), int(oc), int(ph), int(sh), int(dh)+1]


def extract_shapes(df):
    items = df.tolist()

    outputs = []
    for item in items:
        if item.startswith("mb"):
            output = parse_line(item)
            # groups = 1
            output.append(1)
        elif item.startswith("g"):
            groups = item.split("g")[1].split("mb")[0]
            groups_excluded = item.split("mb")[1]
            output = parse_line(item)
            
            output.append(int(groups))
        else:
            assert False, "unsupported prefix: " + item
        outputs.append(output)
    
    print(outputs)
    
    with open('shape.json', 'w', encoding='utf-8') as f:
        json.dump(outputs, f, ensure_ascii=False, indent=4)

def main(file):
    df = preprocess(file)
    
    df = df[df["primitive"] == "convolution"]
    
    problem = df["problem_desc"].drop_duplicates()
    
    # print(problem)
    
    extract_shapes(problem)
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare perf of two files')
    parser.add_argument("-f", "--file_name", default=None, type=str, required=True, help="path to the input onednn log file")
    parser.add_argument('--delimiter', '-d', default='')
    
    args = parser.parse_args()
    
    file_a = args.file_name
    
    main(file_a)