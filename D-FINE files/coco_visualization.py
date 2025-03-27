import argparse
import fiftyone as fo

def main(args):
    # Import the dataset using the provided paths
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.COCODetectionDataset,
        data_path=args.data_path,
        labels_path=args.labels_path,
    )

    # Launch the FiftyOne app with the dataset and limit the view if specified
    session = fo.launch_app(dataset, auto=False)
    session.view = dataset.take(args.take)
    session.show()
    
    # Optionally wait for the session to close before exiting the script
    session.wait()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch FiftyOne app with a COCO Detection dataset."
    )
    parser.add_argument(
        "--data_path", "-i",
        type=str,
        required=True,
        help="Path to the source images directory."
    )
    parser.add_argument(
        "--labels_path", "-l",
        type=str,
        required=True,
        help="Path to the COCO labels JSON file."
    )
    parser.add_argument(
        "--take", "-t",
        type=int,
        default=1000,
        help="Number of images to show in the session (default: 1000)."
    )
    args = parser.parse_args()
    main(args)
