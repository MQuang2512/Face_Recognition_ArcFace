import cv2

# Mở camera (0 là camera mặc định)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể nhận frame")
        break

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
