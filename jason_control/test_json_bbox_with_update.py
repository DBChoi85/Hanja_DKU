import json


# bbox 집합 값을 유지하기 위해 area를 사용함
area = []
filename = 'bboxtojson.json'
with open(filename, "r") as json_file:
    data = json.load(json_file)
    for index, p in enumerate(data['bbox[x1,y1,x2,y2]']):
        area.append(p)

    # 0번쨰 bbox 값을 얻어옴 area[0]
    if area != None:
       c1 = area[0]
       x1 = int(c1[0])
       y1 = int(c1[1])
       x2 = int(c1[2])
       y2 = int(c1[3])
       print ( "%d, %d, %d, %d" %(x1, y1, x2,y2))


# 현재 bbox 값 갱신을 위해 tmp 에 저장함  (UI 에서 영역 갱신을 취소하는 경우 restore을 위해 사용함
tmp = data['bbox[x1,y1,x2,y2]']

# 변경되어진 bbox 값 설정
bbox = list()
bbox = [10, 20, 30, 40]

# 4번재  bbox 가   [10, 20, 30, 40] 로 변경된것을 반영하여 ubboxtojson.json 저장함
data["bbox[x1,y1,x2,y2]"][3] = bbox
jsonFile = open("ubboxtojson.json", "w+")
jsonFile.write(json.dumps(data))
jsonFile.close()
