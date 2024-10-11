import cv2
import numpy as np
import os
import csv
import time

# Function to calculate Euclidean distance using float precision to avoid overflow
def euclidean_distance_float(p1, p2):
    return np.sqrt((float(p1[0]) - float(p2[0])) ** 2 + (float(p1[1]) - float(p2[1])) ** 2)

# Function to match circles between frames based on nearest distances
def match_circles_between_frames(previous_circles, current_circles):
    matched_circles = {}
    used_indices = set()

    for idx, prev_circle in previous_circles.items():
        prev_center = prev_circle['center']
        min_distance = float('inf')
        match_idx = -1

        for i, current_circle in enumerate(current_circles):
            if i in used_indices:
                continue

            curr_center = (current_circle[0], current_circle[1])
            distance = euclidean_distance_float(prev_center, curr_center)

            if distance < min_distance:
                min_distance = distance
                match_idx = i

        if match_idx != -1 :
            matched_circles[idx] = {
                'center': (current_circles[match_idx][0], current_circles[match_idx][1]),
                'radius': current_circles[match_idx][2]
            }
            used_indices.add(match_idx)

    return matched_circles

# Main function to detect and track circles in the video
def detect_and_track_circles(video_path):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video at {video_path}")
        return

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video properties: {frame_count} frames, {fps} FPS, Resolution: {width}x{height}")

    # Create a resizable window and set its size to match the video resolution
    cv2.namedWindow("Circle Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Circle Tracking", width, height)

    # Read the first frame to initialize circles
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect circles in the first frame
    initial_circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=7,
        param1=48,
        param2=14,
        minRadius=4,
        maxRadius=12
    )
    
    if initial_circles is None:
        print("No circles detected in the first frame.")
        return

    # Convert circle data into dictionary format for easy tracking
    initial_circles = np.round(initial_circles[0, :]).astype("int")
    circle_data = {
        i: {'center': (circle[0], circle[1]), 'radius': circle[2], 'initial_center': (circle[0], circle[1])}
        for i, circle in enumerate(initial_circles)
    }

    # Circle trajectories
    circle_trajectories = {i: [(circle[0], circle[1])] for i, circle in enumerate(initial_circles)}

    # Process each frame to track circles
    for frame_num in range(1, frame_count):
        ret, frame = cap.read()
        if not ret:
            print(f"Error reading frame {frame_num}")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        new_circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=7,
            param1=48,
            param2=14,
            minRadius=4,
            maxRadius=12
        )

        if new_circles is not None:
            new_circles = np.round(new_circles[0, :]).astype("int")

            # Match the detected circles with previous frame's circles
            matched_circles = match_circles_between_frames(circle_data, new_circles)

            # Update circle data and trajectories based on matched circles
            for circle_id, circle in matched_circles.items():
                circle_data[circle_id] = circle
                circle_trajectories[circle_id].append(circle['center'])

            # Update positions of unmatched circles (no movement detected)
            for circle_id in circle_data:
                if circle_id not in matched_circles:
                    circle_trajectories[circle_id].append(circle_data[circle_id]['center'])

        # Visualize the tracking (optional)
       
        for circle_id, circle in circle_data.items():
            cv2.circle(frame, circle['center'], circle['radius'], (0, 255, 0), 2)
            #cv2.putText(frame, str(circle_id), circle['center'], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        cv2.imshow("Circle Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        if frame_num % 100 == 0:
            print(f"Processed {frame_num}/{frame_count} frames")
        #"""
    cap.release()
    cv2.destroyAllWindows()

    # Write tracking results to CSV
    output_file = 'circle_tracking_results.csv'
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Circle ID', 'Initial X', 'Initial Y', 'Final X', 'Final Y', 'Total Displacement'])
        
        for circle_id, trajectory in circle_trajectories.items():
            initial_pos = trajectory[0]
            final_pos = trajectory[-1]
            displacement = euclidean_distance_float(initial_pos, final_pos)
            csvwriter.writerow([circle_id, initial_pos[0], initial_pos[1], final_pos[0], final_pos[1], displacement])

    print(f"Circle tracking results have been saved to {output_file}")

if __name__ == "__main__":
    video_path = r'C:\Users\agraw\CircleMovementDetection\VideoNoMov.mp4'
    start_time = time.time()
    detect_and_track_circles(video_path)
    end_time = time.time()
    print(f'Total time taken to run circle detection: {end_time - start_time:.2f} seconds')
