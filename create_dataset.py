import os
import shutil
import cv2
import logging

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


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

while True:

    ret, frame = cap.read()

    cv2.imshow("Main Window", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()