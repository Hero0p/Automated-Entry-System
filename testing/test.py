import cv2
import os

# Load the Haar Cascade model
model_path = 'haarcascade_plate_number.xml'
cascade = cv2.CascadeClassifier(model_path)

# Test dataset paths
test_images_path = 'images'
results = []

# Loop through all test images
for img_name in os.listdir(test_images_path):
    img_path = os.path.join(test_images_path, img_name)
    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Perform detection
    objects = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Save results (filename, objects detected)
    results.append((img_name, len(objects)))

    # Optional: Visualize the detection
    for (x, y, w, h) in objects:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('Detection', image)
    cv2.waitKey(500)  # Show each image for 500ms (adjust as needed)
cv2.destroyAllWindows()

# Print or save results
print(results)
