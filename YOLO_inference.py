from ultralytics import YOLO # imports YOLO class from ultralytics

model = YOLO("models/best.pt") # initializes YOLO object called model by loading weights from file specified

results = model.track("input_data/input_video.mp4") # tracks the input file
# result is a list with length equal to the number of images the model evaluates
#print(results)
print(f'length of list results: {len(results)}')

print(type(results[0])) # original shape of first frame

# for result in results:
#     boxes = result.boxes  # Boxes object for bounding box outputs
#     masks = result.masks  # Masks object for segmentation masks outputs
#     keypoints = result.keypoints  # Keypoints object for pose outputs
#     probs = result.probs  # Probs object for classification outputs
#     obb = result.obb  # Oriented boxes object for OBB outputs
#     #result.show()  # display to screen
#     #result.save(filename="result.mp4")  # save to disk


# print("boxes:")
# for box in result[0].boxes:
#     print(box)