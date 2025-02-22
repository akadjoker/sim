import cv2
import numpy as np
from typing import Tuple, Optional

class TrackController:
    def __init__(self, car):
        self.car = car
        self.autodrive = True
        self.current_speed = 0
        self.steering_angle = 0
        self.track_state = "unknown"  # "straight", "curve_left", "curve_right", "unknown"
        
        # PID Controller parameters
        self.KP = 0.35
        self.KI = 0.005
        self.KD = 1.5
        self.pid_old_error = 0.0
        self.pid_integral = 0.0
        self.pid_need_reset = False
        
        # Track detection parameters
        self.curve_memory = []
        self.curve_memory_size = 5
        
        # Normalized speed settings (-1 to 1)
        self.STRAIGHT_SPEED = 0.8  # 100% of max speed
        self.CURVE_SPEED = 0.4     # 40% of max speed
        self.UNKNOWN_SPEED = 0.3   # 30% of max speed
        
        # Image processing settings
        self.ROI_HEIGHT = 0.3  # Examine bottom 30% of image
        
    def set_speed(self, speed: float) -> None:
        """Set car speed in normalized range -1 to 1"""
        normalized_speed = max(-1.0, min(1.0, speed))
        self.current_speed = normalized_speed
        self.car.set_speed(normalized_speed)
        
    def set_steering_angle(self, angle: float) -> None:
        """Set steering angle in normalized range -1 to 1"""
        # Smooth transitions by limiting steering rate
        max_change = 0.15  # Maximum change per update
        angle_diff = angle - self.steering_angle
        if abs(angle_diff) > max_change:
            angle = self.steering_angle + (max_change if angle_diff > 0 else -max_change)
            
        # Ensure angle is within valid range
        normalized_angle = max(-1.0, min(1.0, angle))
        self.steering_angle = normalized_angle
        self.car.set_steer(normalized_angle)
    
    def detect_center_line(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect the center dashed line of the track"""
        if frame is None:
            return None
            
        # Focus on bottom portion of image
        height, width = frame.shape[:2]
        roi = frame[int(height * (1 - self.ROI_HEIGHT)):, :]
        
        # Convert to HSV for better filtering
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # White/light color detection range for center line
        lower_white = np.array([0, 0, 180])  # Low saturation, high value
        upper_white = np.array([180, 50, 255])  # Any hue, low saturation, high value
        
        # Create mask for white/light colored pixels
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Clean up the mask
        kernel = np.ones((3, 3), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # Filter contours by size and shape
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Filter small noise
            if area < 50:
                continue
                
            # Check for elongated shape (typical of line segments)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = max(w, h) / (min(w, h) + 0.01)  # Avoid division by zero
            
            if aspect_ratio > 2.0:  # Elongated shape
                valid_contours.append(contour)
        
        # If no valid contours found
        if not valid_contours:
            return None
            
        # Sort by y-position (get the lines closer to bottom of image)
        valid_contours.sort(key=lambda c: np.mean([p[0][1] for p in c]), reverse=True)
        
        # Return the best candidate for center line
        return valid_contours[0] if valid_contours else None
    
    def determine_center_error(self, frame: np.ndarray, center_line) -> Optional[float]:
        """Calculate error based on the detected center line"""
        if frame is None or center_line is None:
            return None
            
        height, width = frame.shape[:2]
        image_center_x = width / 2
        
        # Calculate the center of the detected line
        moments = cv2.moments(center_line)
        if moments["m00"] == 0:
            return None
            
        center_x = moments["m10"] / moments["m00"]
        
        # Calculate error (-1 to 1 range)
        error = (center_x - image_center_x) / (width / 2)
        
        # Invert error because we want to steer toward the line, not away
        return -error
    
    def detect_track_curvature(self, center_line, frame_width):
        """Detect if the track is curving left, right, or straight based on center line"""
        if center_line is None or len(center_line) < 5:
            return "unknown"
            
        # Fit a line to the center line points
        points = np.array([p[0] for p in center_line])
        vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        slope = vx / vy if vy != 0 else float('inf')
        
        # Apply smoothing using memory of recent curves
        self.curve_memory.append(slope)
        if len(self.curve_memory) > self.curve_memory_size:
            self.curve_memory.pop(0)
            
        smoothed_slope = np.mean(self.curve_memory)
        
        # Determine track state based on smoothed slope
        if abs(smoothed_slope) < 0.15:  # Near vertical lines indicate straight track
            return "straight"
        elif smoothed_slope > 0:
            return "curve_right"
        else:
            return "curve_left"
            
    def apply_pid(self, error: float) -> float:
        """Apply PID control to the center line error"""
        if self.pid_need_reset:
            self.pid_old_error = error
            self.pid_integral = 0.0
            self.pid_need_reset = False
        
        # Anti-windup mechanism
        if np.sign(error) != np.sign(self.pid_old_error):
            self.pid_integral = 0.0
        
        # Calculate derivative
        derivative = error - self.pid_old_error
        
        # Update integral with windowing
        if -0.3 < self.pid_integral < 0.3:
            self.pid_integral += error
        
        self.pid_old_error = error
        
        # Calculate control output (-1 to 1 range)
        control = (self.KP * error + 
                  self.KI * self.pid_integral + 
                  self.KD * derivative)
        
        return max(-1.0, min(1.0, control))
        
    def adjust_speed_for_track(self):
        """Adjust speed based on track curvature"""
        if self.track_state == "straight":
            target_speed = self.STRAIGHT_SPEED
        elif self.track_state in ["curve_left", "curve_right"]:
            target_speed = self.CURVE_SPEED
        else:
            target_speed = self.UNKNOWN_SPEED
            
        # Smooth speed transitions
        if target_speed > self.current_speed:
            self.set_speed(min(target_speed, self.current_speed + 0.05))
        else:
            self.set_speed(max(target_speed, self.current_speed - 0.05))

    def visualization(self, frame):
        if frame is None:
            print(" Visualization: No Frame")
            return frame
            
        debug_frame = frame.copy()
        height, width = debug_frame.shape[:2]
        
        # Draw ROI line
        roi_y = int(height * (1 - self.ROI_HEIGHT))
        cv2.line(debug_frame, (0, roi_y), (width, roi_y), (0, 255, 0), 2)
        
        # Process image to detect center line
        center_line = self.detect_center_line(frame)
        
        # Draw detected center line
        if center_line is not None:
            cv2.drawContours(debug_frame, [center_line], -1, (0, 255, 255), 2)
            
            # Draw the fitted line for curvature detection
            points = np.array([p[0] for p in center_line])
            vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
            
            # Calculate two points on the line
            lefty = int((-x0 * vy / vx) + y0) if vx != 0 else 0
            righty = int(((width - x0) * vy / vx) + y0) if vx != 0 else height
            
            # Draw the fitted line
            cv2.line(debug_frame, (0, lefty), (width, righty), (255, 0, 0), 2)
        
        # Calculate and draw image center line
        cv2.line(debug_frame, (int(width/2), height), (int(width/2), height - 50), (255, 255, 0), 1)
        
        # Calculate error based on center line
        center_error = self.determine_center_error(frame, center_line)
        
        # Display track state
        if center_line is not None:
            track_state = self.detect_track_curvature(center_line, width)
        else:
            track_state = "unknown"
        
        # Add text for debugging information
        cv2.putText(debug_frame, f"Track: {track_state}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_frame, f"Speed: {self.current_speed:.2f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(debug_frame, f"Steering: {self.steering_angle:.2f}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if center_error is not None:
            cv2.putText(debug_frame, f"Error: {center_error:.2f}", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return debug_frame
            
    def update(self, frame: np.ndarray) -> None:
        if not self.autodrive or frame is None:
            return
            
        # Process image to detect center line
        center_line = self.detect_center_line(frame)
        
        # Calculate target angle based on center line position
        center_error = self.determine_center_error(frame, center_line)
        
        # Detect track curvature
        if center_line is not None:
            self.track_state = self.detect_track_curvature(center_line, frame.shape[1])
        else:
            self.track_state = "unknown"
            
        # Apply PID control if we have a valid center error
        if center_error is not None:
            # Calculate steering angle using PID (already in -1 to 1 range)
            control_output = self.apply_pid(center_error)
            self.set_steering_angle(control_output)
            
            # Adjust speed based on track curvature
            self.adjust_speed_for_track()
        else:
            # Lost track - slow down and reset PID
            self.set_speed(self.current_speed * 0.7)
            self.pid_need_reset = True
            
    def set_autodrive(self, enabled: bool) -> None:
        """Enable or disable autonomous driving"""
        if self.autodrive == enabled:
            return
        self.autodrive = enabled
        print(f"{'Ligado' if enabled else 'Desligado'} piloto automático")
        
        if not enabled:
            # Stop the car when disabling autodrive
            self.set_steering_angle(0)
            self.set_speed(0)