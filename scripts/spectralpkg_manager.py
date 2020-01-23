import os
import argparse
import logging

parser = argparse.ArgumentParser(description='Build or modify a .spectralpkg project.')

parser.add_argument('-m', '--mode', nargs=1, default='add', choices=('add', 'remove'),
                    help='Mode flag for package management. \"add\" for '
                         'artifact addition and \"remove\" for artifact removal. Defaults to add.')

parser.add_argument('artifact_path', nargs='*', type=os.path.abspath,
                    help='Path(s) to artifacts that will be included in the the project package.')

parser.add_argument('-proj_path', '--project_path', nargs='?', default='.', type=os.path.abspath,
                    help='The destination path to create or modify a spectral package. Defaults to working directory.')

parser.add_argument('-proj_id', '--project_id', nargs=1, default='project',
                    help='Optional project identifier and filename.')

parser.add_argument('-a_id', '--artifact_id', nargs='*',
                    help='Optional identifiers for each artifact data file.')

parser.add_argument('-con_m', '--content_mask', nargs='*', type=os.path.abspath,
                    help='Optional path to artifact content masks.')

parser.add_argument('-cla_m', '--class_mask', nargs='*', type=os.path.abspath,
                    help='Optional path to artifact class masks.')

parser.add_argument('-us', '--use_symlink', action='store_true',
                    help='Use this flag to write only a reference to the source data object '
                         'rather than copy it directly. Useful for large source files')

parser.add_argument('-of', '--overwrite_files', action='store_true',
                    help='Specify flag to overwrite files when adding identically '
                         'named artifacts to a spectral package.')

if __name__ == '__main__':
    args = parser.parse_args()
    assert args.artifact_id is None or len(args.artifact_id) == len(args.artifact_path), \
        "Number of artifact id's must match the number of artifacts"
    assert args.content_mask is None or len(args.content_mask) == len(args.artifact_path), \
        "Number of content masks must match the number of artifacts"
    assert args.class_mask is None or len(args.class_mask) == len(args.artifact_path), \
        "Number of class masks must match the number of artifacts"

    package_name = os.path.join(args.project_path, args.project_id + ".spectralpkg")
    if not os.path.exists(package_name):
        if args.mode == "add":
            os.mkdir(package_name)
            for top_level_folder in ("Artifacts", "Transforms", "Analysis"):
                os.makedirs(os.path.join(package_name, top_level_folder))

            for artifact_path in args.artifact_path:
                destination = os.path.join(package_name, "Artifacts", os.path.basename(artifact_path))
                if os.path.exists(destination) and not args.overwirte_files:
                    raise ValueError("Error, artifact with that file name alreadt exists")
                if args.use_symlink:
                    os.symlink(artifact_path, os.path.join(package_name, "Artifacts"))
                else:
                    # TODO, write file copy routine

        else:
            raise FileNotFoundError("Error, the specified spectral package does not exist, "
                                    "if you are trying to make it use add mode")

