import cv2
import numpy as np

VIDEO_PATH = "road.mp4"

prev_left_lane = None
prev_right_lane = None

def region_of_interest(img):
    h, w = img.shape[:2]
    mask = np.zeros_like(img)

    polygon = np.array([[
        (int(w * 0.10), h),
        (int(w * 0.43), int(h * 0.63)),
        (int(w * 0.57), int(h * 0.63)),
        (int(w * 0.90), h)
    ]], dtype=np.int32)

    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)

def average_lane_line(lines, img_height):
    if len(lines) == 0:
        return None

    x_points = []
    y_points = []

    for x1, y1, x2, y2 in lines:
        x_points.extend([x1, x2])
        y_points.extend([y1, y2])

    if len(x_points) < 2:
        return None

    fit = np.polyfit(y_points, x_points, 1)  # x = m*y + b

    y1 = img_height
    y2 = int(img_height * 0.75)

    x1 = int(fit[0] * y1 + fit[1])
    x2 = int(fit[0] * y2 + fit[1])

    return (x1, y1, x2, y2)

def smooth_line(current_line, prev_line, alpha=0.2):
    if current_line is None:
        return prev_line
    if prev_line is None:
        return current_line

    smoothed = []
    for c, p in zip(current_line, prev_line):
        smoothed.append(int(alpha * c + (1 - alpha) * p))
    return tuple(smoothed)

def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    masked_edges = region_of_interest(edges)

    lines = cv2.HoughLinesP(
        masked_edges,
        1,
        np.pi / 180,
        threshold=50,
        minLineLength=60,
        maxLineGap=100
    )

    left_lines = []
    right_lines = []

    h, w = frame.shape[:2]

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)

            if abs(slope) < 0.5 or abs(slope) > 2.5:
                continue

            if slope < 0 and x1 < w * 0.52 and x2 < w * 0.52:
                left_lines.append((x1, y1, x2, y2))
            elif slope > 0 and x1 > w * 0.48 and x2 > w * 0.48:
                right_lines.append((x1, y1, x2, y2))

    left_lane = average_lane_line(left_lines, h)
    right_lane = average_lane_line(right_lines, h)

    return masked_edges, left_lane, right_lane

def draw_results(frame, left_lane, right_lane):
    result = frame.copy()
    overlay = result.copy()
    h, w = result.shape[:2]

    if left_lane is not None:
        x1, y1, x2, y2 = left_lane
        cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 6)

    if right_lane is not None:
        x1, y1, x2, y2 = right_lane
        cv2.line(result, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 6)

    if left_lane is not None and right_lane is not None:
        pts = np.array([[
            (int(left_lane[0]), int(left_lane[1])),
            (int(left_lane[2]), int(left_lane[3])),
            (int(right_lane[2]), int(right_lane[3])),
            (int(right_lane[0]), int(right_lane[1]))
        ]], dtype=np.int32)

        cv2.fillPoly(overlay, pts, (0, 80, 0))
        result = cv2.addWeighted(overlay, 0.35, result, 0.65, 0)

    car_center = int(w // 2)
    cv2.line(result, (car_center, int(h)), (car_center, int(h * 0.75)), (0, 0, 255), 3)

    if left_lane is not None and right_lane is not None:
        left_bottom_x = int(left_lane[0])
        right_bottom_x = int(right_lane[0])
        lane_center = int((left_bottom_x + right_bottom_x) / 2)

        lane_center = int(0.7 * car_center + 0.3 * lane_center)

        cv2.line(result, (lane_center, int(h)), (lane_center, int(h * 0.75)), (255, 0, 0), 3)

        offset = int(lane_center - car_center)
        cv2.putText(
            result,
            f"Offset: {offset}px",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (255, 255, 255),
            2
        )

    return result

def main():
    global prev_left_lane, prev_right_lane

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {VIDEO_PATH}")
        input("Press Enter to exit...")
        return

    print("Video opened successfully. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        edges, left_lane, right_lane = detect_lanes(frame)

        left_lane = smooth_line(left_lane, prev_left_lane, alpha=0.25)
        right_lane = smooth_line(right_lane, prev_right_lane, alpha=0.25)

        prev_left_lane = left_lane
        prev_right_lane = right_lane

        result = draw_results(frame, left_lane, right_lane)

        cv2.imshow("Edges", edges)
        cv2.imshow("Lane Detection PRO", result)

        key = cv2.waitKey(25) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()