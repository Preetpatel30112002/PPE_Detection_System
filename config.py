class Config:
    MODEL_PATH = "D:/PES/PPE_Detection_System/runs/detect/train/weights/best.pt"
    VIDEO_PATH = "D:/PES/PPE_Detection_System/test videos/Sample_video3.mp4"
    CONFIDENCE_THRESHOLD = 0.1
    IOU_THRESHOLD = 0.45
    PIXEL_PERCENTAGE_THRESHOLD = 80

    PERSON_CLASS = 1
    HELMET_CLASS = 0
    BOOT_CLASS = 2

    COLORS = {
        'GREEN_BOX': (0, 255, 0),
        'RED_BOX': (0, 0, 255),
        'BLUE_BOX': (255, 0, 0),
        'PERSON_BOX': (255, 255, 0)
    }

    SNAPSHOTS_DIR = "D:/PES/PPE_Detection_System/snapshots"

    OUTPUT_VIDEO_CODEC = 'XVID'

    BOX_THICKNESS = 2
    FONT_SCALE = 0.7
    FONT_THICKNESS = 2
    PADDING = 20