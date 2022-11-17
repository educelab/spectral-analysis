#!/bin/bash

#SBATCH -A col_seales_uksr
#SBATCH --mail-type=END
#SBATCH --job-name=spectral-analysis
#SBATCH --output=spectral-analysis_%A_%a_out.txt

# Make rclone available on the container and tell system where to look
#SBATCH --export=SINGULARITY_BIND='/share/singularity/bin',SINGULARITYENV_PREPEND_PATH='/share/singularity/bin'

container=${SPEC_AN_CONTAINER:-"${PROJECT}/seales_uksr/containers/spectral-analysis.sif"}
overlay=${SPEC_AN_OVERLAY:-"spectral-analysis.overlay"}

# Make sure overlay files exist
if ! test -f "${overlay}" ; then
  echo "Creating ${overlay}"
  dd if=/dev/zero of="${overlay}" bs=1M count=500 && mkfs.ext3 -F "${overlay}"
fi

module load ccs/singularity

time singularity run --overlay ${overlay} ${container} "$@"