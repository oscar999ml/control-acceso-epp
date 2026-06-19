import cv2
import time

for backend_name, backend in [("DEFAULT", None), ("DSHOW", cv2.CAP_DSHOW)]:
    for idx in [5, 0, 1]:
        try:
            if backend is not None:
                cap = cv2.VideoCapture(idx, backend)
            else:
                cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                ret, frame = cap.read()
                print(f"Backend {backend_name} index {idx}: opened={cap.isOpened()}, frame={ret}, shape={frame.shape if ret else 'N/A'}")
                cap.release()
            else:
                print(f"Backend {backend_name} index {idx}: NOT opened")
        except Exception as e:
            print(f"Backend {backend_name} index {idx}: error {e}")

cv2.destroyAllWindows()
