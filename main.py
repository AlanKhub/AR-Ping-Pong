import math
import cv2
import mediapipe as mp
import time
import random

score_1 = 0
score_2 = 0
time1 = 0
time2 = 0
class Ball:
    def __init__(self):
        self.speedx = 15
        self.speedy = 15
        self.x = 100
        self.y = 100


ball = Ball()


def shortest_distance_to_line(x0, y0, x12, y12, xc, yc, r):
    # Step 1: Calculate the perpendicular distance from the circle center to the line
    numerator = abs((y12 - y0) * xc - (x12 - x0) * yc + x12 * y0 - y12 * x0)
    denominator = math.sqrt((y12 - y0) ** 2 + (x12 - x0) ** 2)

    d_perp = numerator / denominator

    # Step 2: Calculate the dot product to check if the perpendicular lies within the line segment
    # Vector AB = (x12 - x0, y12 - y0) and vector AC = (xc - x0, yc - y0)
    AB_dot_AC = (x12 - x0) * (xc - x0) + (y12 - y0) * (yc - y0)
    AB_dot_AB = (x12 - x0) ** 2 + (y12 - y0) ** 2

    # Calculate the projection factor
    t = AB_dot_AC / AB_dot_AB

    if 0 <= t <= 1:
        # The perpendicular projection falls on the segment, so use the perpendicular distance
        if d_perp <= r:
            return 0  # Circle intersects the segment
        return d_perp - r  # Otherwise, return the distance minus the radius
    else:
        # The perpendicular projection falls outside the segment, so calculate the distance to the nearest endpoint
        # Calculate the distance to the nearest endpoint (x0, y0) or (x12, y12)
        dist_to_p1 = math.sqrt((xc - x0) ** 2 + (yc - y0) ** 2)
        dist_to_p2 = math.sqrt((xc - x12) ** 2 + (yc - y12) ** 2)
        # Return the shortest distance to the endpoints minus the radius
        return min(dist_to_p1, dist_to_p2) - r

class HandTrackingDynamic:

    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.__mode__ = mode
        self.__maxHands__ = maxHands
        self.__detectionCon__ = detectionCon
        self.__trackCon__ = trackCon
        self.handsMp = mp.solutions.hands
        self.hands = self.handsMp.Hands(max_num_hands=self.__maxHands__,
                                        min_detection_confidence=self.__detectionCon__,
                                        min_tracking_confidence=self.__trackCon__)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findFingers(self, frame, draw=False):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        return frame

    def findPosition(self, frame, draw=True):
        global ball, time1
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                # Extract only the positions of points 0 and 12 (Wrist and Middle Finger Tip)
                x0, y0 = int(handLms.landmark[0].x * frame.shape[1]), int(
                    handLms.landmark[0].y * frame.shape[0])  # Wrist (Point 0)
                x12, y12 = int(handLms.landmark[8].x * frame.shape[1]), int(
                    handLms.landmark[8].y * frame.shape[0])  # Middle Finger Tip (Point 12)
                centerX = int((x0 + x12) / 2)
                centerY = int((y0 + y12) / 2)
                sigma_distance = shortest_distance_to_line(x0, y0, x12, y12, ball.x, ball.y, 50)
                if sigma_distance < 5 and time.time() - time1 > 0.5:
                    ball.speedx *= -1
                    time1 = time.time()
                try:
                    angle = math.atan((y12 - y0) / (x12 - x0)) * 57.29
                except ZeroDivisionError:
                    angle = 90
                # print(angle)
                # Draw only the two points (Wrist and Middle Finger Tip)
                if draw:
                    cv2.circle(frame, (x0, y0), 5, (0, 0, 255), cv2.FILLED)  # Red for wrist (Point 0)
                    cv2.circle(frame, (x12, y12), 5, (0, 255, 0), cv2.FILLED)  # Green for middle finger tip (Point 12)
                    cv2.circle(frame, (centerX, centerY), 5, (255, 200, 200), -1)
                    # Draw a line between point 0 and point 12
                    cv2.line(frame, (x0, y0), (x12, y12), (255, 0, 0), 2)  # Blue line between points 0 and 12

        return frame


def main():
    global  ball, score_1, score_2
    ctime = 0
    ptime = 0
    game_over = 0
    cap = cv2.VideoCapture(0)
    detector = HandTrackingDynamic()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 960)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    ret, frame = cap.read()
    if game_over != 1:
        while game_over != 1:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            # Find landmarks and draw points 0 and 12, along with the line between them
            frame = detector.findFingers(frame)
            frame = detector.findPosition(frame, draw=True)
            #print(ball.speedx)
            ctime = time.time()
            fps = 1 / (ctime - ptime)
            ptime = ctime
            ball.x += ball.speedx
            ball.y += ball.speedy
            if ball.x >= 1280:
                score_1 += 1
                ball.x = random.randint(300, 980)
                ball.y = random.randint(60, 900)
                if ball.speedx > 0:
                    ball.speedx += random.randint(-2, 5)
                else:
                    ball.speedx -= random.randint(-2, 5)
                if ball.speedy > 0:
                    ball.speedy += random.randint(-2, 5)
                else:
                    ball.speedy -= random.randint(-2, 5)
            if ball.x <= 0:
                score_2 += 1
                ball.x = random.randint(300, 980)
                ball.y = random.randint(60, 900)
                if ball.speedx > 0:
                    ball.speedx += random.randint(-2, 5)
                else:
                    ball.speedx -= random.randint(-2, 5)
                if ball.speedy > 0:
                    ball.speedy += random.randint(-2, 5)
                else:
                    ball.speedy -= random.randint(-2, 5)
            if score_1 >= 10 or score_2 >= 10:
                game_over = 1
            ball.x %= 1280
            if ball.y < 50 or ball.y > 910:
                ball.speedy *= -1
            # Show FPS on the frame
            cv2.putText(frame,  f'Score: {score_1} : {int(score_2)}', (460, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            cv2.circle(frame, (ball.x, ball.y), 50, (50, 255, 175), -1)

            # Display the frame
            cv2.imshow('frame', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC to exit
                break

        cv2.rectangle(frame, (0, 0), (1280, 960), (0, 0, 0), -1)
        cv2.putText(frame, f'Game Over', (200, 480), cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 255), 15)
        cv2.imshow('frame', frame)

        cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
