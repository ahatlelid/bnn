
import sys
import json

s_file = sys.argv[1]


with open(s_file) as f:
    data = f.read()
      
js = json.loads(data)
  
print("Data type after reconstruction : ", type(js))
print(js)