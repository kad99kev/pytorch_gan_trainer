import os
import shutil
import cv2
import logging

from tqdm import tqdm

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


def create_dataset(LIMIT, IMAGE_SIZE, DATASET_PATH):

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

    images_captured = 0
    is_capturing = False
    length = 0

    while True:

        ret, frame = cap.read()

        frame = cv2.flip(frame, 1)

        display = frame.copy()
        frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))

        cv2.imshow("Main Window", display)
        cv2.imshow("Virtual Window", frame)

        # Create new label
        if cv2.waitKey(1) & 0xFF == ord('c'):
            label = input('Enter label name: ')
            label_path = DATASET_PATH + f'/{label}/'
            if not os.path.isdir(label_path):
                os.mkdir(label_path)
                is_capturing = True
                pbar = tqdm(total=LIMIT)

        if is_capturing:
            image_path = label_path + f'{images_captured}.jpg'
            cv2.imwrite(image_path, frame)
            images_captured += 1
            pbar.update()
            if images_captured == LIMIT:
                pbar.close()
                is_capturing = False
                images_captured = 0
                logging.info(f'Finished creating dataset for label: {label}')
                
        
        # Quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()