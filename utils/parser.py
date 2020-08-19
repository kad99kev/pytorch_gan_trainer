import argparse

def parse_arguments():

    parser = argparse.ArgumentParser(description='SelfieGAN')

    parser.add_argument('--mode', type=str, action='store', help='Whether to create dataset or train model.', default='create', choices=['create', 'train'])

    parser.add_argument(
        '--image_size', type=int, action='store', help='Size of the image dataset.', default=256
    )

    parser.add_argument(
        '--limit', type=int, action='store', help='Number of images to click per class.', default=100
    )

    parser.add_argument(
        '--output_dir', type=str, action='store', help='Location to store dataset.', default='./dataset'
    )
    
    return parser.parse_args()
