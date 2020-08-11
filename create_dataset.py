import os
import shutil
import cv2
import logging

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

LIMIT = 5
IMAGE_SIZE = 512
DATASET_PATH = './dataset'

# Make directory for dataset
if os.path.isdir(DATASET_PATH):
    logging.warning("Dataset directory already exists! Overwrite?")
    while True:
        choice = input('Enter Y for Yes and n for no: ').strip()
        if choice == 'Y':
            # Overwrite directory
            logging.info(f'Creating new dataset at location: {DATASET_PATH}')
            shutil.rmtree(DATASET_PATH)
            os.mkdir(DATASET_PATH)
            break
        elif choice == 'n':
            # Do nothing
            raise OSError('Cannot create new dataset! Provide an alternative path instead.')
        else:
            print('Please enter Y or n!')
else:
    logging.info(f'Creating new directory at location: {DATASET_PATH}')
    os.mkdir(DATASET_PATH)


logging.info("Loading Camera...")
cap = cv2.VideoCapture(0)
logging.info("Camera Loaded!")

is_capturing = False
images_captured = 0
length = 0

while True:

    ret, frame = cap.read()

    frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))

    cv2.imshow("Main Window", frame)

    # Create new label
    if cv2.waitKey(1) & 0xFF == ord('c'):
        label = input('Enter label name: ')
        label_path = DATASET_PATH + f'/{label}/'
        if os.path.isdir(label_path):
            os.mkdir(label_path)
            is_capturing = True
        else:
            length = len([name for name in os.listdir(label_path) if os.path.isfile(name)])
            print('Add to dataset')

    if is_capturing:
        image_path = label_path + f'{images_captured + length}.jpg'
        cv2.imwrite(image_path, frame)
        image_captured += 1
        if image_captured == LIMIT:
            is_capturing = False
    
    # Quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()