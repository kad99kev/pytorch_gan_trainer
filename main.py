from utils.create_dataset import create_dataset
from utils.parser import parse_arguments

def main(args):

    if args.mode == 'create':
        create_dataset(args.limit, args.image_size, args.output_dir)
    else:
        pass

if __name__ == '__main__':
    main(parse_arguments())