import os
import argparse

import numpy as np
import imageio

from spectral_analysis.spectral_io import SpectralPackageManager

# Initialize the top level parser with common flags, set it up for the various mode flags as sub parsers
parser = argparse.ArgumentParser(description='Build or modify a .spctrl (Spectral Package) project')
parser.add_argument('-proj_id', '--project_id', nargs='?', default='project', type=str,
                    help='Optional project identifier and filename. Defaults to "project"')
parser.add_argument('-proj_path', '--project_path', nargs='?', default='.', type=os.path.abspath,
                    help='The destination path to create or modify a spectral package. Defaults to working directory')
# TODO, make multiple sub parser objects which share or don't share flags related to id values
subparsers = parser.add_subparsers(required=True, help='Mode flag for package management', dest="mode")

# The create sub parser handles commands for creating spectral packages
create_parser = subparsers.add_parser("create", help='Construct a package')

# TODO, add support for adding/removing transforms and pipelines
# The add sub parser handles adding new samples, existing transforms, and pipeline specifiers
add_parser = subparsers.add_parser("add", help='Add a sample, transform, or pipeline to a package')
add_parser.add_argument('sample_path', nargs='*', type=os.path.abspath,
                        help='Path(s) to samples that will be included in the the project package')
add_parser.add_argument('-s_id', '--sample_id', nargs='*',
                        help='Identifiers for each sample data file. Required for all or none, defaults to "sample_n"')
# TODO, wrap masks into one type then have mask purpose defined later in the pipeline
add_parser.add_argument('-con_m', '--content_mask', nargs='*', type=os.path.abspath,
                        help='Optional path to sample content masks. Required for all or none')
add_parser.add_argument('-cla_m', '--class_mask', nargs='*', type=os.path.abspath,
                        help='Optional path to sample class masks. Required for all or none')
add_parser.add_argument('-cs', '--copy_source', action='store_true',
                        help='Use this flag to store only a reference to the source data object '
                             'rather than copy it directly. Useful for large source files')
add_parser.add_argument('-of', '--overwrite_files', action='store_true',
                        help='Specify flag to overwrite files when adding identically '
                        'named samples, transforms, or pipelines to a spectral package')

# The remove sub parser handles removing samples, transforms, and pipelines
remove_parser = subparsers.add_parser("remove", help="Remove a sample, transform, or pipeline to a package")
remove_parser.add_argument('-s_id', '--sample_id', nargs='+',
                           help='Identifiers for each sample to remove')

# The list sub parser controls listing contents of the spectral package
list_parser = subparsers.add_parser("list", help="List all sample, transforms, or pipelines in a package")
list_parser.add_argument("-w", "--which", default="all", choices=("all", "samples", "transforms", "pipelines"),
                         nargs="?", help="Specify what type of package contents to list")

# The view sub parser handles visualization of samples, transforms, or pipelines
view_parser = subparsers.add_parser("view", help="Visualize a sample, transform, or pipeline")
view_parser.add_argument("-p1", "--point_1", nargs=2, default=(0, 0), help="x y coordinates of  to extract")
view_parser.add_argument("-p2", "--point_2", nargs=2, default=(np.inf, np.inf))
view_parser.add_argument("-od", "--output_dir", default=".", type=os.path.abspath)
view_parser.add_argument("-sid", "--sample_ids", nargs="+", type=os.path.abspath)
view_parser.add_argument("-of", "--output_filetype", default="png")
view_parser.add_argument("-sb", "--spectral_band", type=float, required=True)

# TODO, implement run parser
# The run parser handles executing computational graphs defined in pipelines
run_parser = subparsers.add_parser("run", help="Run a pipeline stored in the spectral package")


def main():
    args = parser.parse_args()
    filepath = os.path.join(args.project_path, args.project_id + '.spctrl')

    if not os.path.exists(filepath) and args.mode != "create":
        raise ValueError("Error, no spectral package by that name exists")

    elif args.mode == "create":
        if os.path.exists(filepath):
            raise ValueError("Error, a spectral package already exists with that name")
        else:
            SpectralPackageManager.make_package(args.project_path, args.project_id)

    else:
        manager = SpectralPackageManager(filepath)

        if args.mode == "add":
            # Assertion checks to ensure number of sample ids/sample masks match number of samples
            assert args.sample_id is None or len(args.sample_id) == len(args.sample_path), \
                "Number of sample id's must match the number of samples"
            assert args.content_mask is None or len(args.content_mask) == len(args.sample_path), \
                "Number of content masks must match the number of samples"
            assert args.class_mask is None or len(args.class_mask) == len(args.sample_path), \
                "Number of class masks must match the number of samples"

            for i, sample_path in enumerate(args.sample_path):
                sample_id = None if args.sample_id is None else args.sample_id[i]
                cont_mask = None if args.content_mask is None else args.content_mask[i]
                class_mask = None if args.class_mask is None else args.class_mask[i]
                manager.add_sample(sample_path, sample_id=sample_id, overwrite=args.overwrite_files,
                                   copy_path=args.copy_source, cont_mask=cont_mask, class_mask=class_mask)

        elif args.mode == "remove":
            for id_val in args.sample_id:
                manager.remove_sample(id_val)

        elif args.mode == "list":
            if args.which in ("all", "samples"):
                print("Samples:")
                for sample_id in manager.sample_ids:
                    print("\t", sample_id)

            if args.which in ("all", "transforms"):
                print("Transforms:")
                for transform_id in manager.transform_ids:
                    print("\t", transform_id)

            if args.which in ("all", "pipelines"):
                print("Pipelines:")
                for pipeline_id in manager.pipeline_ids:
                    print("\t", pipeline_id)

        # TODO, expand view to open files etc
        elif args.mode == "view":
            x1, y1 = args.point_1
            x2, y2 = args.point_2

            for sample_id in args.sample_ids:
                save_name = f"{sample_id}.{args.output_filetype}"

                data_handler = manager.get_sample_handler(sample_id)

                # Do this gross mess because the envi parser counts wavelength metadata as
                # extra info without a dedicated parsing section in the segment in which it is initially read in
                image_bands = data_handler.metadata["extra data"][
                    "wavelength"].replace(",", "").strip("{}\n").split("\n")
                image_bands = np.array(image_bands, dtype=np.float)
                band_difference = np.abs(image_bands - args.spectral_band)
                image_index = np.argmin(band_difference)

                print(f"Extracting image with closest wavelength of {image_bands[image_index]}")

                image = data_handler.io.get_volume_chunk((x1, x2), (y1, y2), (image_index, image_index + 1))

                imageio.imsave(os.path.join(args.output_dir, save_name), image)

        elif args.mode == "run":
            # TODO, implement pipeline handler which will process and execute scripts chaining data and transforms
            pass


if __name__ == '__main__':
    main()
