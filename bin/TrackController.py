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
        # self.KP = 0.3
        # self.KI = 0.005
        # self.KD = 1.5
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
        
    def detect_track_edges(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Detect both inner and outer yellow edges of the track"""
        if frame is None:
            return None, None
            
        # Focus on bottom portion of image
        height, width = frame.shape[:2]
        roi = frame[int(height * (1 - self.ROI_HEIGHT)):, :]
        
        # Convert to HSV for better color filtering
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # lower = np.array([15, 150, 150])#laranja
        # upper = np.array([25, 255, 255])
        
        # lower = np.array([70, 0, 180])#white
        # upper = np.array([80, 50, 255])


        lower = np.array([20, 100, 100])
        upper = np.array([40, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower, upper)
        
        # Clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, None
            
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        # Try to identify inner and outer track boundaries
        if len(contours) >= 2:
            # Assume the two largest contours are the track boundaries
            c1 = contours[0]
            c2 = contours[1]
            
            # Determine which is left and which is right edge
            c1_x = np.mean([p[0][0] for p in c1])
            c2_x = np.mean([p[0][0] for p in c2])
            
            left_edge = c1 if c1_x < c2_x else c2
            right_edge = c2 if c1_x < c2_x else c1
            
            return left_edge, right_edge
        elif len(contours) == 1:
            # Only one edge detected, try to determine if it's left or right
            edge = contours[0]
            edge_x = np.mean([p[0][0] for p in edge])
            
            if edge_x < roi.shape[1] / 2:
                return edge, None  # Left edge
            else:
                return None, edge  # Right edge
        
        return None, None
        
    def determine_center_line(self, frame: np.ndarray, left_edge, right_edge) -> Optional[float]:
        """Calculate the center line position from track edges"""
        if frame is None:
            return None
            
        height, width = frame.shape[:2]
        
        if left_edge is not None and right_edge is not None:
            # Both edges detected - calculate center line
            left_points = [p[0] for p in left_edge]
            right_points = [p[0] for p in right_edge]
            
            left_x = np.mean([p[0] for p in left_points])
            right_x = np.mean([p[0] for p in right_points])
            
            center_x = (left_x + right_x) / 2
        elif left_edge is not None:
            # Only left edge detected - estimate center
            left_points = [p[0] for p in left_edge]
            left_x = np.mean([p[0] for p in left_points])
            # Estimate track width as 1/3 of the image width
            estimated_track_width = width / 3
            center_x = left_x + estimated_track_width / 2
        elif right_edge is not None:
            # Only right edge detected - estimate center
            right_points = [p[0] for p in right_edge]
            right_x = np.mean([p[0] for p in right_points])
            # Estimate track width as 1/3 of the image width
            estimated_track_width = width / 3
            center_x = right_x - estimated_track_width / 2
        else:
            return None
            
        # Calculate error from image center (normalized -1 to 1)
        center_error = (center_x - width / 2) / (width / 2)
        return center_error
        
    def detect_track_curvature(self, left_edge, right_edge, frame_width):
        """Detect if the track is curving left, right, or straight"""
        if left_edge is None and right_edge is None:
            return "unknown"
            
        # Calculate the average slope of the edges
        slopes = []
        
        if left_edge is not None and len(left_edge) >= 2:
            # Fit a line to the left edge
            left_points = np.array([p[0] for p in left_edge])
            vx, vy, x0, y0 = cv2.fitLine(left_points, cv2.DIST_L2, 0, 0.01, 0.01)
            slope_left = vx / vy if vy != 0 else float('inf')
            slopes.append(slope_left)
            
        if right_edge is not None and len(right_edge) >= 2:
            # Fit a line to the right edge
            right_points = np.array([p[0] for p in right_edge])
            vx, vy, x0, y0 = cv2.fitLine(right_points, cv2.DIST_L2, 0, 0.01, 0.01)
            slope_right = vx / vy if vy != 0 else float('inf')
            slopes.append(slope_right)
        
        if not slopes:
            return "unknown"
            
        avg_slope = np.mean(slopes)
        
        # Apply smoothing using memory of recent curves
        self.curve_memory.append(avg_slope)
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
            print(" Visualation: No Frame")
            return frame
            
        debug_frame = frame.copy()
        height, width = debug_frame.shape[:2]
        
        # Draw ROI line
        roi_y = int(height * (1 - self.ROI_HEIGHT))
        cv2.line(debug_frame, (0, roi_y), (width, roi_y), (0, 255, 0), 2)
        
        # Process image to detect track
        left_edge, right_edge = self.detect_track_edges(frame)
        
        # Draw detected edges
        if left_edge is not None:
            cv2.drawContours(debug_frame, [left_edge], -1, (255, 0, 0), 2)
        if right_edge is not None:
            cv2.drawContours(debug_frame, [right_edge], -1, (0, 0, 255), 2)
        
        # Calculate and draw center line
        center_error = self.determine_center_line(frame, left_edge, right_edge)
        if center_error is not None:
            center_x = int(width / 2 + center_error * (width / 2))
            cv2.line(debug_frame, (center_x, height), (center_x, height - 50), (0, 255, 255), 2)
            cv2.line(debug_frame, (int(width/2), height), (int(width/2), height - 50), (255, 255, 0), 1)
        
        # Display track state
        if left_edge is not None or right_edge is not None:
            track_state = self.detect_track_curvature(left_edge, right_edge, width)
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
            
        # Process image to detect track
        left_edge, right_edge = self.detect_track_edges(frame)
        
        # Calculate target angle
        center_error = self.determine_center_line(frame, left_edge, right_edge)
        
        # Detect track curvature
        if left_edge is not None or right_edge is not None:
            self.track_state = self.detect_track_curvature(left_edge, right_edge, frame.shape[1])
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