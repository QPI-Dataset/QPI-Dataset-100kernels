# Quasiparticle Interference Kernel Extraction with Variational Autoencoders via Latent Alignment
Quasiparticle interference (QPI) imaging is a powerful tool for probing electronic structures in quantum materials, but extracting the single-scatterer QPI pattern (i.e., the kernel) from a multi-scatterer image remains a fundamentally ill-posed inverse problem, because many different kernels can combine to produce almost the same observed image, and noise or overlaps further obscure the true signal. Existing solutions to this extraction problem rely on manually zooming into small local regions with isolated single-scatterers. This is infeasible for real cases where scattering conditions are too complex. In this work, we propose the first AI-based framework for QPI kernel extraction, which models the space of physically valid kernels and uses this knowledge to guide the inverse mapping. We introduce a two-step learning strategy that decouples kernel representation learning from observation-to-kernel inference. In the first step, we train a variational autoencoder to learn a compact latent space of scattering kernels. In the second step, we align the latent representation of QPI observations with those of the pre-learned kernels using a dedicated encoder. This design enables the model to infer kernels robustly under complex, entangled scattering conditions. We construct a diverse and physically realistic QPI dataset comprising 100 unique kernels and evaluate our method against a direct one-step baseline. Experimental results demonstrate that our approach achieves significantly higher extraction accuracy, improved generalization to unseen kernels.  To further validate its effectiveness, we also apply the method to real QPI data from Ag and FeSe samples, where it reliably extracts meaningful kernels under complex scattering conditions.

<img width="808" height="649" alt="model_architecture" src="https://github.com/user-attachments/assets/cf250ba9-99fd-49f5-9985-474e084eb6b8" />
<img width="694" height="190" alt="infer1" src="https://github.com/user-attachments/assets/450bff73-fd9c-49e7-90bf-e97d676c9d1e" />

Scatterer Dataset - 64×64 Resolution, 400×400 Å FOV
This dataset contains synthetic 2D images generated from scattering kernels of atomic-scale features. The images are 64×64 pixels in resolution, corresponding to a 400×400 Å (Angstroms) field of view (FOV). Each image is constructed from contributions of up to 20 scatterers positioned in real space. (**To download the dataset, please click the tag**)

📁 File Format Overview
Each file in this dataset contains a set of rows with structured numerical data. The contents are organized as follows:

Row 1: Metadata
Column 1: Resolution (always 64 for this dataset)

Column 2: Maximum number of scatterers (always 20 for this dataset)

Row 2: Real-space pixel coordinates
Columns 1–64: X-axis coordinates in real space (in Å), linearly spanning from -200 Å to 200 Å

Example: pixel 1 is at x = -200 Å, pixel 64 is at x = 200 Å

Row 3: Kernel image and scatterer location
Columns 1–4096: Flattened 64×64 kernel image (1D array of pixel intensities)

Columns 4097–4136: X and Y coordinates (Å) of up to 20 scatterers (40 values total: x1, y1, x2, y2, ..., x20, y20)

For the kernel, only a single scatterer is present at (x=0, y=0); the rest are set to 1000 to indicate no scatterer

Rows 4 and beyond: Full images and scatterer positions
Each subsequent row represents a full image constructed from multiple scatterers.

Columns 1–4096: Flattened 64×64 image

Columns 4097–4136: Real-space coordinates of up to 20 scatterers that contributed to the image

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
