import cv2
import numpy as np
import mediapipe as mp
import time
import math

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Initialize drawing parameters
draw_color = (0, 0, 255)  # Red color by default
brush_thickness = 15
eraser_thickness = 50

# Create canvas
canvas = np.zeros((480, 640, 3), np.uint8)

# Previous coordinates for drawing
prev_x, prev_y = 0, 0

# For smoothing
smoothing_factor = 0.5
smooth_x, smooth_y = 0, 0

# Button states
button_hover = None

# Define color buttons
buttons = [
    {"name": "RED", "color": (0, 0, 255), "rect": (40, 1, 140, 65), "text_color": (255, 255, 255)},
    {"name": "GREEN", "color": (0, 255, 0), "rect": (160, 1, 260, 65), "text_color": (255, 255, 255)},
    {"name": "BLUE", "color": (255, 0, 0), "rect": (280, 1, 380, 65), "text_color": (255, 255, 255)},
    {"name": "ERASER", "color": (0, 0, 0), "rect": (400, 1, 500, 65), "text_color": (255, 255, 255)},
    {"name": "CLEAR", "color": (255, 255, 255), "rect": (520, 1, 620, 65), "text_color": (0, 0, 0)}
]

# Function to find hand landmarks with improved stability
def find_hand_landmarks(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    landmarks = []
    
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]  # Get the first hand
        mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        for id, lm in enumerate(hand_landmarks.landmark):
            h, w, c = img.shape
            cx, cy = int(lm.x * w), int(lm.y * h)
            landmarks.append((cx, cy))
    
    return img, landmarks

# Create color selection buttons with hover effect
def create_buttons(img, hover_button=None):
    for button in buttons:
        x1, y1, x2, y2 = button["rect"]
        color = button["color"]
        name = button["name"]
        text_color = button["text_color"]
        
        # Draw button with highlight if hovered
        if hover_button == name:
            # Draw highlight border
            cv2.rectangle(img, (x1-3, y1-3), (x2+3, y2+3), (255, 255, 0), 3)
            # Draw button with slightly brighter color
            bright_color = tuple(min(c + 50, 255) for c in color)
            cv2.rectangle(img, (x1, y1), (x2, y2), bright_color, -1)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        
        # Center text
        text_size = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        text_x = x1 + (x2 - x1 - text_size[0]) // 2
        text_y = y1 + (y2 - y1 + text_size[1]) // 2
        cv2.putText(img, name, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)
    
    return img


def is_point_in_button(point, button):
    x, y = point
    x1, y1, x2, y2 = button["rect"]
    return x1 < x < x2 and y1 < y < y2


def smooth_coordinates(current_x, current_y, prev_x, prev_y, smoothing_factor):
    return int(prev_x + smoothing_factor * (current_x - prev_x)), int(prev_y + smoothing_factor * (current_y - prev_y))

# Main function
def main():
    global prev_x, prev_y, draw_color, canvas, button_hover, smooth_x, smooth_y
    
    
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Width
    cap.set(4, 480)  # Height
    
    
    prev_time = 0
    
    # Mode tracking
    selection_mode = False
    drawing_mode = False
    previous_drawing_mode = False  # Track previous drawing mode state
    
    # For stabilizing the drawing
    points_history = []
    max_history = 5
    
    while True:
        # Read frame from webcam
        success, img = cap.read()
        if not success:
            break
            
        
        img = cv2.flip(img, 1)
        
        # Find hand landmarks
        img, landmarks = find_hand_landmarks(img)
        
        # Reset hover state
        hover_button = None
        
       
        if landmarks:
           
            index_finger_tip = landmarks[8]
            middle_finger_tip = landmarks[12]
            
            
            index_up = landmarks[8][1] < landmarks[6][1]  
            middle_up = landmarks[12][1] < landmarks[10][1]  
            
            
            selection_mode = index_up and middle_up
            drawing_mode = index_up and not middle_up
            
          
            if drawing_mode and not previous_drawing_mode:
                prev_x, prev_y = 0, 0
                smooth_x, smooth_y = 0, 0
                points_history = []
            
            if selection_mode:
                
                for button in buttons:
                    if is_point_in_button(index_finger_tip, button):
                        hover_button = button["name"]
                        
                       
                        thumb_tip = landmarks[4]
                        distance = math.sqrt((thumb_tip[0] - index_finger_tip[0])**2 + 
                                            (thumb_tip[1] - index_finger_tip[1])**2)
                        
                        if distance < 40:  
                            if button["name"] == "RED":
                                draw_color = (0, 0, 255)
                            elif button["name"] == "GREEN":
                                draw_color = (0, 255, 0)
                            elif button["name"] == "BLUE":
                                draw_color = (255, 0, 0)
                            elif button["name"] == "ERASER":
                                draw_color = (0, 0, 0)
                            elif button["name"] == "CLEAR":
                                canvas = np.zeros((480, 640, 3), np.uint8)
                
                prev_x, prev_y = 0, 0
                smooth_x, smooth_y = 0, 0
                points_history = []
                
               
                cv2.circle(img, index_finger_tip, 15, (255, 255, 0), cv2.FILLED)
                cv2.circle(img, middle_finger_tip, 15, (255, 255, 0), cv2.FILLED)
                cv2.line(img, index_finger_tip, middle_finger_tip, (255, 255, 0), 3)
            
            
            elif drawing_mode:
             
                if smooth_x == 0 and smooth_y == 0:
                    smooth_x, smooth_y = index_finger_tip
                else:
                    smooth_x, smooth_y = smooth_coordinates(
                        index_finger_tip[0], index_finger_tip[1], 
                        smooth_x, smooth_y, 
                        smoothing_factor
                    )
                
              
                points_history.append((smooth_x, smooth_y))
                if len(points_history) > max_history:
                    points_history.pop(0)
                
                
                if len(points_history) > 0:
                    avg_x = sum(p[0] for p in points_history) // len(points_history)
                    avg_y = sum(p[1] for p in points_history) // len(points_history)
                    current_point = (avg_x, avg_y)
                else:
                    current_point = (smooth_x, smooth_y)
                
                
                cv2.circle(img, current_point, 15, draw_color, cv2.FILLED)
                
              
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = current_point
                else:
                  
                    if draw_color == (0, 0, 0):  # Eraser
                        cv2.line(canvas, (prev_x, prev_y), current_point, draw_color, eraser_thickness)
                    else:
                        cv2.line(canvas, (prev_x, prev_y), current_point, draw_color, brush_thickness)
                
              
                prev_x, prev_y = current_point
            
            # Update previous drawing mode state
            previous_drawing_mode = drawing_mode
        else:
           
            prev_x, prev_y = 0, 0
            smooth_x, smooth_y = 0, 0
            points_history = []
            previous_drawing_mode = False
        
     
        img = create_buttons(img, hover_button)
        
        # Calculate FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time
        
        #  FPS
        cv2.putText(img, f'FPS: {int(fps)}', (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        
        mode_text = "SELECTION MODE" if selection_mode else "DRAWING MODE" if drawing_mode else "NO MODE"
        cv2.putText(img, mode_text, (220, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
     
        color_name = "ERASER" if draw_color == (0, 0, 0) else "RED" if draw_color == (0, 0, 255) else "GREEN" if draw_color == (0, 255, 0) else "BLUE"
        cv2.putText(img, f"Color: {color_name}", (450, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_color, 2)
        
      
        img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, img_inv)
        img = cv2.bitwise_or(img, canvas)
        
        cv2.putText(img, "Index finger: Draw", (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, "Index + Middle fingers: Select", (220, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, "Pinch to select color", (450, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
      
        cv2.imshow("Hand Drawing Canvas", img)
        
       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
   
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()