from ultralytics import YOLO

model = YOLO("yolo11n.pt")

result = model.predict("data/TennisMatch1.mp4", conf=0.2,save=True)
print(result)
# result2 = model.predict("data/streetPic.png.webp")
# print(result2)

print("boxes:")
for box in result[0].boxes:
    print(box)