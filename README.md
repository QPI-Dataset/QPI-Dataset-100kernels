# QPI-Dataset-100kernels
Scatterer Dataset - 64×64 Resolution, 100×100 Å FOV
This dataset contains synthetic 2D images generated from scattering kernels of atomic-scale features. The images are 64×64 pixels in resolution, corresponding to a 100×100 Å (Angstroms) field of view (FOV). Each image is constructed from contributions of up to 100 scatterers positioned in real space. (**The files are in the tag**)

📁 File Format Overview
Each file in this dataset contains a set of rows with structured numerical data. The contents are organized as follows:

Row 1: Metadata
Column 1: Resolution (always 64 for this dataset)

Column 2: Maximum number of scatterers (always 100 for this dataset)

Row 2: Real-space pixel coordinates
Columns 1–64: X-axis coordinates in real space (in Å), linearly spanning from -50 Å to 50 Å

Example: pixel 1 is at x = -50 Å, pixel 64 is at x = 50 Å

Row 3: Kernel image and scatterer location
Columns 1–4096: Flattened 64×64 kernel image (1D array of pixel intensities)

Columns 4097–4296: X and Y coordinates (Å) of up to 100 scatterers (200 values total: x1, y1, x2, y2, ..., x100, y100)

For the kernel, only a single scatterer is present at (x=0, y=0); the rest are set to 1000 to indicate no scatterer

Rows 4 and beyond: Full images and scatterer positions
Each subsequent row represents a full image constructed from multiple scatterers.

Columns 1–4096: Flattened 64×64 image

Columns 4097–4296: Real-space coordinates of up to 100 scatterers that contributed to the image

Positions not used are filled with 1000, indicating absence (outside FOV)

📌 Notes
The 1000 placeholder is used to indicate missing or unused scatterer coordinates; it lies well outside the FOV and does not affect the image.

By storing scatterer positions as real-space coordinates rather than binary masks, storage is significantly reduced.

The kernel row (Row 3) provides the contribution of a single central scatterer and serves as a basis for understanding how each individual scatterer contributes to the final image.

🧪 Applications
This dataset can be used for:

Training and validating machine learning models for inverse scattering problems

Benchmarking algorithms for scatterer detection and image reconstruction

Visualizing contributions of individual and collective atomic-scale features
