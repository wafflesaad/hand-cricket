import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import random


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

latest_result = None
def on_result(result, output_image, timestamp_ms):
    global latest_result
    latest_result = result

# Initialize the hand landmarker with the live stream mode and the on_result callback.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1, running_mode=vision.RunningMode.LIVE_STREAM,  result_callback=on_result )
detector = vision.HandLandmarker.create_from_options(options)

# indices for hand landmarks to be drawn
imp_indices = [0,2,4,5,8,9,12,13,16,17,20]


score = 0
bot_score = 0
bat = True

last_time = time.time()
bot_play = -1

enemy_score = 0
end_game = False

while True:
    success, img = cap.read()

    if not success:
        break

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(mp.ImageFormat.SRGB, data=imgRGB)
    timestamp = int(time.time() * 1000)
    detector.detect_async(mp_img, timestamp)


    if latest_result:
        for hand in latest_result.hand_landmarks:

            # extract relevant landmarks
            wrist, thumb_base, thumb_tip, index_base, index_tip, middle_base, middle_tip, ring_base, ring_tip, pinky_base, pinky_tip = points = [hand[i] for i in imp_indices]

            h,w,_ = img.shape

            if bat:
                cv2.putText(img,"Batting", (int(w/2), h-30), cv2.FONT_HERSHEY_SIMPLEX,1.0, color=(0,0,0), thickness=2)
            elif not bat and end_game:
                if score < enemy_score:
                    cv2.putText(img,"Lost", (int(w/2), h-30), cv2.FONT_HERSHEY_SIMPLEX,1.0, color=(0,0,0), thickness=2)
                elif score > enemy_score:
                    cv2.putText(img,"Won", (int(w/2), h-30), cv2.FONT_HERSHEY_SIMPLEX,1.0, color=(0,0,0), thickness=2)
                else:
                    cv2.putText(img,"Tie", (int(w/2), h-30), cv2.FONT_HERSHEY_SIMPLEX,1.0, color=(0,0,0), thickness=2)
            else:
                cv2.putText(img,"Balling", (int(w/2), h-30), cv2.FONT_HERSHEY_SIMPLEX,1.0, color=(0,0,0), thickness=2)

            # Counting extended fingers
            fingers = []

            if thumb_tip.x < thumb_base.x:
                fingers.append(1)
            else:
                fingers.append(0)


            for tip, knuckle in [(index_tip, index_base), (middle_tip, middle_base), (ring_tip, ring_base), (pinky_tip, pinky_base)]:

                if tip.y < knuckle.y:
                    fingers.append(1)
                else:
                    fingers.append(0)
                



            for lm in points:
                
                cx,cy = int(lm.x*w), int(lm.y*h)
                cv2.circle(img, (cx,cy), 5, (0,255,0),-1)


            # lines for hand connections
            cv2.line(img,(int(wrist.x*w),int(wrist.y*h)), (int(thumb_base.x*w), int(thumb_base.y*h)), (255,0,0), 1)
            cv2.line(img,(int(thumb_base.x*w), int(thumb_base.y*h)), (int(thumb_tip.x*w), int(thumb_tip.y*h)), (255,0,0), 1)
            cv2.line(img,(int(wrist.x*w),int(wrist.y*h)), (int(index_base.x*w), int(index_base.y*h)), (255,0,0), 1)
            cv2.line(img,(int(index_base.x*w), int(index_base.y*h)), (int(index_tip.x*w), int(index_tip.y*h)), (255,0,0), 1)
            cv2.line(img,(int(wrist.x*w),int(wrist.y*h)), (int(middle_base.x*w), int(middle_base.y*h)), (255,0,0), 1)   
            cv2.line(img,(int(middle_base.x*w), int(middle_base.y*h)), (int(middle_tip.x*w), int(middle_tip.y*h)), (255,0,0), 1)
            cv2.line(img,(int(wrist.x*w),int(wrist.y*h)), (int(ring_base.x*w), int(ring_base.y*h)), (255,0,0), 1)
            cv2.line(img,(int(ring_base.x*w), int(ring_base.y*h)), (int(ring_tip.x*w), int(ring_tip.y*h)), (255,0,0), 1)
            cv2.line(img,(int(wrist.x*w),int(wrist.y*h)), (int(pinky_base.x*w), int(pinky_base.y*h)), (255,0,0), 1)
            cv2.line(img,(int(pinky_base.x*w), int(pinky_base.y*h)), (int(pinky_tip.x*w), int(pinky_tip.y*h)), (255,0,0), 1)

            count = 6 if fingers[0] == 1 and all(x == 0 for x in fingers[1:]) else sum(fingers)

            

            if time.time() - last_time >= 5:

                round_count = count

                bot_play = random.randint(1,6)

                if bot_play == round_count and bat:
                    bat = False
                elif bot_play == round_count and not bat:
                    end_game = True

                if bat:                  
                    # bot_play = 2

                    # bot_score += bot_play
                

                    # out 
                    
                    score += round_count

                    cv2.putText(img,str(count), (int(w/2), 30), cv2.FONT_HERSHEY_SIMPLEX,1.0, color=(0,0,0), thickness=2)
                    cv2.putText(img,"Score: " + str(score), (30, 30), cv2.FONT_HERSHEY_SIMPLEX,1.0, color=(0,0,0), thickness=2)
                    cv2.putText(img, str(bot_play), (w - 30, 30), cv2.FONT_HERSHEY_SIMPLEX,1.0, color=(0,0,0), thickness=2)
                else:

                    enemy_score += bot_play
                    cv2.putText(img,str(enemy_score), (int(w/2), 30), cv2.FONT_HERSHEY_SIMPLEX,1.0, color=(0,0,0), thickness=2)
                    cv2.putText(img,"Score: " + str(score), (30, 30), cv2.FONT_HERSHEY_SIMPLEX,1.0, color=(0,0,0), thickness=2)
                    cv2.putText(img, str(bot_play), (w - 30, 30), cv2.FONT_HERSHEY_SIMPLEX,1.0, color=(0,0,0), thickness=2)

                last_time = time.time()

            if bat:
                cv2.putText(img,str(count), (int(w/2), 30), cv2.FONT_HERSHEY_SIMPLEX,1.0, color=(0,0,0), thickness=2)
                cv2.putText(img,"Score: " + str(score), (30, 30), cv2.FONT_HERSHEY_SIMPLEX,1.0, color=(0,0,0), thickness=2)
                cv2.putText(img, str(bot_play), (w - 30, 30), cv2.FONT_HERSHEY_SIMPLEX,1.0, color=(0,0,0), thickness=2)
            else:
                cv2.putText(img,str(count), (int(w/2), 30), cv2.FONT_HERSHEY_SIMPLEX,1.0, color=(0,0,0), thickness=2)
                cv2.putText(img,"Score: " + str(enemy_score), (30, 30), cv2.FONT_HERSHEY_SIMPLEX,1.0, color=(0,0,0), thickness=2)
                cv2.putText(img, str(bot_play), (w - 30, 30), cv2.FONT_HERSHEY_SIMPLEX,1.0, color=(0,0,0), thickness=2)

            cv2.putText(img,str(int(time.time() - last_time)), (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX,1.0, color=(0,0,0), thickness=2)


            





    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
detector.close()