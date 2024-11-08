
import cv2

# Update the path to the custom Haar Cascade XML file for Indian number plates
harcascade = "C:\\Users\\nisha\\Desktop\\coding\\Big Projects\\Automated entry system\\model\\haarcascade_russian_plate_number.xml"


ip_address = "http://192.0.0.4:8080/video"  # or /shot.jpg for testing
cap = cv2.VideoCapture(ip_address)

cap.set(3, 640)  # Width
cap.set(4, 480)  # Height

min_area = 500

count = 0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    # Load the Indian number plate cascade
    plate_cascade = cv2.CascadeClassifier(harcascade)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAYw)

    # Adjust parameters if needed (1.1 and 5 are good for initial testing)
    plates = plate_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in plates:
        area = w * h
        if area > min_area:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
            img_roi = img[y: y+h , x: x+w]
            cv2.imshow("ROI", img_roi)


        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite("plates/scaned_img" + ".jpg", img_roi)
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 2)
        cv2.imshow("Result", img)
        cv2.waitKey(500)
        count += 1


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
