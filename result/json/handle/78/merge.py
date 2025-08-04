import json


json1="../../0413/a/new_key_merged_a2.json"
json2="../../0413/a/new_key_merged_a3.json"
json3=""

json_r="../../0413/a/result_first.json"
with open(json1,"r") as f:
    d1=json.load(f)
with open(json2, "r") as f:
    d2 = json.load(f)
# with open(json3, "r") as f:
#     d3 = json.load(f)

d={**d1,**d2}

with open(json_r,"w")as f:
    json.dump(d,f,indent=4)
