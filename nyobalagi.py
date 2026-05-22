import cv2

#Read File
PATH = r'D:\uni - 4th sem\uni 4th\CV\FinalProjectCV_Kelompok03\dataset\Cv Malam.mp4'
cap = cv2.VideoCapture(PATH)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.resize (frame, (1280, 720))
    cv2.imshow('Frame', frame) #read frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()