import cv2
import numpy as np
from rich.live import Live
from rich.table import Table
from rich.console import Console

console = Console()

def estimate_distance(face_width_pixels, known_width_cm=15.0, focal_length=500):
    """Estimate distance to face based on its pixel width."""
    if face_width_pixels > 0:
        return (known_width_cm * focal_length) / face_width_pixels
    return None

def main():
    # Load face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        console.print("[bold red]Error:[/bold red] Could not open camera")
        return

    # Background subtractor for detecting movement
    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    with Live(refresh_per_second=10) as live:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Face detection
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            face_status = "[red]No[/red]"
            face_distance = "-"
            if len(faces) > 0:
                face_status = "[bold green]FACE DETECTED[/bold green]"
                (x, y, w, h) = faces[0]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Face", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                distance = estimate_distance(w)
                if distance:
                    face_distance = f"{distance:.1f} cm"
                    cv2.putText(frame, face_distance, (x, y + h + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # Moving object detection
            fg_mask = back_sub.apply(frame)
            _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            object_count = 0
            object_details = []
            for cnt in contours:
                if cv2.contourArea(cnt) > 500:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    object_count += 1
                    object_details.append(f"({x},{y}) {w}x{h}")

            # Build dashboard
            table = Table(title="[bold cyan]Live Object Tracking Dashboard[/bold cyan]")
            table.add_column("Section", style="cyan", no_wrap=True)
            table.add_column("Status / Data", style="magenta")

            # CAMERA FEED section
            table.add_row("[bold yellow]CAMERA FEED[/bold yellow]", "Streaming from webcam...")

            # OBJECT STATUS section
            table.add_row("[bold yellow]OBJECT STATUS[/bold yellow]", f"Detected: {object_count} moving objects")
            table.add_row("Object Details", "\n".join(object_details) if object_details else "-")

            # ENVIRONMENT STATUS section
            table.add_row("[bold yellow]ENVIRONMENT STATUS[/bold yellow]", f"Face: {face_status}")
            table.add_row("Face Distance", face_distance)

            live.update(table)

            # Show camera feeds
            cv2.imshow("Face + Background Tracking", frame)
            cv2.imshow("Foreground Mask", fg_mask)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
