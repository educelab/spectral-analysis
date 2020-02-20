import os
import argparse

from spectral_data_io import SpectralPackageManager

parser = argparse.ArgumentParser(description='Build or modify a .spctrl (Spectral Package) project.')

parser.add_argument('-m', '--mode', nargs=1, default='view', choices=('create', 'add', 'remove', 'run', 'view'),
                    help='Mode flag for package management. '
                         '\"create\" for package construction.\n'
                         '\"run\" for execution of a computational pipeline.\n'
                         '\"add\" for sample, transform, or analysis pipeline addition.\n'
                         '\"remove\" for sample, transform, or analysis pipeline removal.\n'
                         'Defaults to view.')

parser.add_argument('sample_path', nargs='*', type=os.path.abspath,
                    help='Path(s) to samples that will be included in the the project package.')

parser.add_argument('-proj_path', '--project_path', nargs='?', default='.', type=os.path.abspath,
                    help='The destination path to create or modify a spectral package. Defaults to working directory.')

parser.add_argument('-proj_id', '--project_id', nargs='?', default='project', type=str,
                    help='Optional project identifier and filename.')

parser.add_argument('-a_id', '--artifact_id', nargs='*',
                    help='Optional identifiers for each artifact data file.')

parser.add_argument('-con_m', '--content_mask', nargs='*', type=os.path.abspath,
                    help='Optional path to sample content masks.')

parser.add_argument('-cla_m', '--class_mask', nargs='*', type=os.path.abspath,
                    help='Optional path to sample class masks.')

parser.add_argument('-cs', '--copy_source', action='store_true',
                    help='Use this flag to store only a reference to the source data object '
                         'rather than copy it directly. Useful for large source files.')

parser.add_argument('-of', '--overwrite_files', action='store_true',
                    help='Specify flag to overwrite files when adding identically '
                         'named samples, transforms, or pipelines to a spectral package.')

# TODO, add support for adding/removing transforms and pipelines

if __name__ == '__main__':
    args = parser.parse_args()
    assert args.artifact_id is None or len(args.artifact_id) == len(args.artifact_path), \
        "Number of sample id's must match the number of samples"
    assert args.content_mask is None or len(args.content_mask) == len(args.artifact_path), \
        "Number of content masks must match the number of samples"
    assert args.class_mask is None or len(args.class_mask) == len(args.artifact_path), \
        "Number of class masks must match the number of samplesls"

    filepath = os.path.join(args.project_path, args.project_id + '.spectralpkg')

    if os.path.exists(filepath) or args.mode == 'add':
        manager = SpectralPackageManager(args.project_path, args.project_id)
    else:
        raise FileNotFoundError("Error, the specified spectral package does not exist, "
                                "if you are trying to generate it use the 'add' mode flag")

    for i, artifact_path in enumerate(args.artifact_path):
        art_id = None if args.artifact_id is None else args.artifact_id[i]
        cont_mask = None if args.content_mask is None else args.content_mask[i]
        class_mask = None if args.class_mask is None else args.class_mask[i]

        if args.mode == "add":
            manager.add_sample(artifact_path, overwrite=args.overwrite_files, use_sym=args.use_symlink, art_id=art_id)

        else:
            manager.remove_sample(art_id)

