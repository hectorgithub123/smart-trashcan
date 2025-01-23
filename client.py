import cv2
import socket
import pickle
import struct
import time
from picamera2 import Picamera2
import numpy as np
from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory

class TrashDetectionClient:
    def __init__(self, server_ip='192.168.67.91', server_port=8000):
        # Initialize camera
        self.camera = Picamera2()
        self.camera.configure(self.camera.create_preview_configuration(
            main={"format": 'RGB888', "size": (640, 480)}))
        self.camera.start()
        
        # Initialize servos
        factory = PiGPIOFactory()
        self.normal_servo = AngularServo(18,
                                       min_pulse_width=0.0006,
                                       max_pulse_width=0.0023,
                                       pin_factory=factory)
        
        self.medical_servo = AngularServo(12,
                                        min_pulse_width=0.0006,
                                        max_pulse_width=0.0023,
                                        pin_factory=factory)
        
        # Set initial position (locked)
        self.normal_servo.angle = 0
        self.medical_servo.angle = 0

        
        # Initialize socket client
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((server_ip, server_port))
        print("Connected to server")

    def send_frame(self, frame):
        try:
            # Ensure frame is correct format
            frame = cv2.resize(frame, (640, 480))
            
            # Keep original BGR format for YOLO
            _, buffer = cv2.imencode('.jpg', frame)
            data = pickle.dumps(np.array(buffer))
            
            # Send frame size and data
            size = struct.pack('>L', len(data))
            self.client_socket.sendall(size)
            self.client_socket.sendall(data)
            
            # Receive result
            result = self.client_socket.recv(1024).decode()
            return result
            
        except Exception as e:
            print(f"Error sending frame: {str(e)}")
            print(f"Frame shape: {frame.shape if frame is not None else 'None'}")
            return "error"

    def run(self):
        try:
            while True:
                # Capture frame
                frame = self.camera.capture_array()
                
                # Display frame
                cv2.imshow('Client View', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                # Send frame and get result
                result = self.send_frame(frame)
                print(f"Detected: {result}")

                self.rotate_servo(result)
                
                # Add delay to control frame rate
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("Stopping client...")
        finally:
            cv2.destroyAllWindows()
            self.camera.stop()
            self.client_socket.close()

    def rotate_servo(self, result):
        if result == "normal trash":
            # Open normal trash compartment
            self.normal_servo.angle = 90
            time.sleep(3)  # Keep open for 3 seconds
            self.normal_servo.angle = 0  # Return to locked position
            
        elif result == "medical trash":
            # Open medical trash compartment
            self.medical_servo.angle = -90
            time.sleep(3)  # Keep open for 3 seconds
            self.medical_servo.angle = 0  # Return to locked position

# Start the client
if __name__ == "__main__":
    client = TrashDetectionClient()
    client.run()