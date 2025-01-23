from ultralytics import YOLO
import socket
import cv2
import numpy as np
import struct
import pickle


"""
YOLO Object Detection Implementation
Using Ultralytics YOLOv11 Model

Citations:
-------------
@software{yolo11_ultralytics,
  author = {Glenn Jocher and Jing Qiu},
  title = {Ultralytics YOLO11},
  version = {11.0.0},
  year = {2024},
  url = {https://github.com/ultralytics/ultralytics},
  orcid = {0000-0001-5950-6979, 0000-0002-7603-6750, 0000-0003-3783-7069},
  license = {AGPL-3.0}
}
"""


class TrashDetectionServer:
    def __init__(self, host='192.168.67.91', port=8000):
        # Initialize YOLO model
        self.model = YOLO("detectmodel.pt")
        
        # Initialize socket server
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((host, port))
        self.server_socket.listen(1)
        print(f"Server listening on port {port}")
        print("YOLO model loaded successfully")

    def receive_frame(self, client_socket):
        try:
            client_socket.settimeout(5.0)
            
            # Get frame size
            size_data = client_socket.recv(4)
            if not size_data:
                print("No size data received")
                return None
                
            data_size = struct.unpack('>L', size_data)[0]
            print(f"Expected frame size: {data_size} bytes")
            
            # Receive frame data
            data = b''
            while len(data) < data_size:
                chunk = client_socket.recv(min(4096, data_size - len(data)))
                if not chunk:
                    print("Connection broken while receiving frame")
                    return None
                data += chunk
            
            # Unpack and decode frame
            frame_data = pickle.loads(data)
            frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
            print(f"Frame received successfully, shape: {frame.shape}")
            return frame
            
        except socket.timeout:
            print("Timeout while receiving frame")
            return None
        except Exception as e:
            print(f"Error receiving frame: {str(e)}")
            return None

    def send_frame(self, frame):
        try:
            # Resize and encode frame
            frame = cv2.resize(frame, (640, 480))
            _, encoded = cv2.imencode('.jpg', frame)
            data = pickle.dumps(encoded)
            
            # Send size and data
            self.client_socket.sendall(struct.pack('>L', len(data)))
            self.client_socket.sendall(data)
            
            return self.client_socket.recv(1024).decode()
            
        except Exception as e:
            print(f"Send error: {str(e)}")
            return "error"

    def process_frame(self, frame):
        try:
            # Validate frame
            if frame is None or not isinstance(frame, np.ndarray):
                print("Invalid frame format")
                return "invalid frame"
                
            # Ensure frame has correct shape and type
            if len(frame.shape) != 3:
                print(f"Invalid frame dimensions: {frame.shape}")
                return "invalid frame"
                
            # Run YOLO detection
            results = self.model.predict(
                source=frame,
                conf=0.5,
                show=False
            )
            
            # Draw detection results on frame
            if len(results) > 0:
                # Get the first result
                result = results[0]
                # Plot the detection boxes
                annotated_frame = result.plot()
                # Show the frame
                cv2.imshow('YOLO Detection', annotated_frame)
                cv2.waitKey(1)
                
                # Process detection results
                boxes = result.boxes
                if len(boxes) > 0 and boxes.cls is not None and len(boxes.cls) > 0:
                    class_id = int(boxes.cls[0])
                    return "medical trash" if class_id == 0 else "normal trash"   
            return "no trash detected"
            
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return "error processing frame"

    def run(self):
        try:
            print("Waiting for client connection...")
            while True:
                client_socket, addr = self.server_socket.accept()
                print(f"Client connected from {addr}")
                
                try:
                    while True:
                        frame = self.receive_frame(client_socket)
                        if frame is None:
                            break
                            
                        result = self.process_frame(frame)
                        client_socket.send(result.encode())
                        
                        # Check for quit
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            raise KeyboardInterrupt
                            
                finally:
                    client_socket.close()
                    
        except KeyboardInterrupt:
            print("Server stopping...")
        finally:
            cv2.destroyAllWindows()
            self.server_socket.close()

if __name__ == "__main__":
    server = TrashDetectionServer()
    server.run()


