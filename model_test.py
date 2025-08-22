
from ultralytics import YOLO
import cv2

# 🔹 Load your trained model
model = YOLO("/home/softsuave/Downloads/best.pt")

# 🔹 Run inference (save=True will auto-save annotated image to runs/detect/predict/)
results = model("/home/softsuave/Downloads/pipe_image.jpeg", save=True)

# 🔹 Get annotated image (numpy array BGR)
annotated_img = results[0].plot()

# 🔹 Show results
cv2.imshow("YOLO Prediction", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 🔹 Save manually if you want custom path
cv2.imwrite("output.jpg", annotated_img)
print("✅ Inference complete, saved as output.jpg")

