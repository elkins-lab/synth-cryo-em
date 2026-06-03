from typing import Any

import gemmi
import mrcfile
import numpy as np
import numpy.typing as npt


def generate_density_map(
    input_path: str,
    resolution: float,
    grid_spacing: float | None = None,
    use_bfactors: bool = False,
    margin: float | None = None,
) -> tuple[gemmi.FloatGrid, npt.NDArray[np.float64]]:
    """
    Generate a density map from an atomic model file (PDB, mmCIF, BCIF) using gemmi.
    If use_bfactors is True, use atomic B-factors for local resolution.
    """
    st = gemmi.read_structure(input_path)
    # If grid_spacing is not provided, use a rule of thumb (resolution / 3 or 4)
    if grid_spacing is None:
        grid_spacing = resolution / 3.0

    # Get all atomic positions
    positions_list: list[list[float]] = []
    for model in st:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    positions_list.append(atom.pos.tolist())

    if not positions_list:
        raise ValueError("No atoms found in structure")

    positions = np.array(positions_list)
    if margin is None:
        margin = resolution * 2.0

    min_pos = positions.min(axis=0) - margin
    max_pos = positions.max(axis=0) + margin
    size = max_pos - min_pos

    st_shifted = st.clone()
    # Ensure spacegroup is P1 for the synthetic box to avoid incorrect symmetry applications
    st_shifted.spacegroup_hm = "P1"
    cell = gemmi.UnitCell(size[0], size[1], size[2], 90, 90, 90)
    st_shifted.cell = cell

    for model in st_shifted:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom.pos.x -= min_pos[0]
                    atom.pos.y -= min_pos[1]
                    atom.pos.z -= min_pos[2]

    # Map the atoms to the grid using DensityCalculatorE
    calc = gemmi.DensityCalculatorE()
    calc.d_min = resolution

    # Calculate sampling rate to match grid_spacing
    # spacing = d_min / (2 * rate)  => rate = d_min / (2 * spacing)
    calc.rate = resolution / (2.0 * grid_spacing)

    # Initialize grid
    calc.set_grid_cell_and_spacegroup(st_shifted)

    # If use_bfactors is True, we use the atomic B-factors.
    # Gemmi's DensityCalculatorE uses atomic B-factors by default
    # when calling put_model_density_on_grid.
    # However, we can add a constant "base" blur to represent resolution.
    # resolution (d) relates to B-factor roughly by B = 8 * pi^2 * (d/2)^2 = 2 * pi^2 * d^2
    # But gemmi also uses d_min as a cutoff.

    if not use_bfactors:
        # If not using B-factors, we set them all to 0 and use a global blur
        # equivalent to the target resolution.
        for model in st_shifted:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        atom.b_iso = 0.0
        # Set a global blur to match the target resolution
        # A common heuristic is B = 8 * res^2 for synthetic maps
        calc.blur = 8.0 * resolution**2

    calc.initialize_grid()
    if len(st_shifted) > 0:
        calc.put_model_density_on_grid(st_shifted[0])

    return calc.grid, min_pos


def add_gaussian_noise(data: npt.NDArray[np.float64], snr: float) -> npt.NDArray[np.float64]:
    """
    Add Gaussian noise to the data based on desired SNR.
    """
    signal_power = np.mean(data**2)
    noise_power = signal_power / snr
    noise = np.random.normal(0, np.sqrt(noise_power), data.shape)
    return data + noise


def apply_ctf(
    data: npt.NDArray[np.float64],
    voxel_size: tuple[float, float, float],
    defoc: float = 2.0,
    cs: float = 2.7,
    voltage: float = 300,
    amplitude_contrast: float = 0.1,
    b_factor: float = 0.0,
) -> npt.NDArray[np.float64]:
    """
    Apply a simple Contrast Transfer Function (CTF) to the 3D data.
    defoc: defocus in micrometers
    cs: spherical aberration in mm
    voltage: acceleration voltage in kV
    b_factor: envelope function B-factor
    """
    # Constants
    wl = 12.26 / np.sqrt(voltage * 1000 + 0.9784 * voltage**2)  # wavelength in Angstroms
    cs_a = cs * 1e7  # cs in Angstroms
    defoc_a = defoc * 10000  # defocus in Angstroms

    nz, ny, nx = data.shape
    # Frequencies
    kz = np.fft.fftfreq(nz, d=voxel_size[0])
    ky = np.fft.fftfreq(ny, d=voxel_size[1])
    kx = np.fft.fftfreq(nx, d=voxel_size[2])

    Kz, Ky, Kx = np.meshgrid(kz, ky, kx, indexing="ij")
    k2 = Kz**2 + Ky**2 + Kx**2

    # Phase shift
    chi = np.pi * wl * k2 * (defoc_a - 0.5 * wl**2 * k2 * cs_a)

    # CTF
    ctf = -(np.sqrt(1 - amplitude_contrast**2) * np.sin(chi) + amplitude_contrast * np.cos(chi))

    # Envelope function
    if b_factor > 0:
        envelope = np.exp(-b_factor * k2 / 4.0)
        ctf *= envelope

    # Apply in Fourier domain
    data_f = np.fft.fftn(data)
    data_f *= ctf
    return np.real(np.fft.ifftn(data_f))


def compute_fsc(
    data1: npt.NDArray[Any],
    data2: npt.NDArray[Any],
    voxel_size: tuple[float, float, float],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Compute the Fourier Shell Correlation (FSC) between two 3D maps.
    Returns frequencies and correlation values.
    """
    assert data1.shape == data2.shape

    # Fourier transforms in float64 for precision
    f1 = np.fft.fftn(data1.astype(np.float64))
    f2 = np.fft.fftn(data2.astype(np.float64))

    # Cross-spectral density
    cross = f1 * np.conj(f2)
    p1 = np.real(f1 * np.conj(f1))
    p2 = np.real(f2 * np.conj(f2))

    # Calculate radial bins
    nz, ny, nx = data1.shape
    kz = np.fft.fftfreq(nz, d=voxel_size[0])
    ky = np.fft.fftfreq(ny, d=voxel_size[1])
    kx = np.fft.fftfreq(nx, d=voxel_size[2])

    Kz, Ky, Kx = np.meshgrid(kz, ky, kx, indexing="ij")
    k = np.sqrt(Kz**2 + Ky**2 + Kx**2)

    # Flatten everything
    k = k.ravel()
    cross = cross.ravel()
    p1 = p1.ravel()
    p2 = p2.ravel()

    # Sort by frequency
    idx = np.argsort(k)
    k_sorted = k[idx]
    cross_sorted = cross[idx]
    p1_sorted = p1[idx]
    p2_sorted = p2[idx]

    # Binning
    n_bins = min(nx, ny, nz) // 2
    bins = np.linspace(0, k_sorted.max(), n_bins + 1)

    fsc = []
    freqs = []

    for i in range(n_bins):
        mask = (k_sorted >= bins[i]) & (k_sorted < bins[i + 1])
        if np.any(mask):
            c_bin = cross_sorted[mask]
            p1_bin = p1_sorted[mask]
            p2_bin = p2_sorted[mask]

            # Sum of cross power and individual powers (using float64 accumulation)
            sum_cross = np.sum(c_bin)
            sum_p1 = np.sum(p1_bin)
            sum_p2 = np.sum(p2_bin)

            # FSC is real part of cross correlation / sqrt(power1 * power2)
            num = np.real(sum_cross)
            den = np.sqrt(sum_p1 * sum_p2)

            if den > 0:
                # Clamp to [-1, 1] to avoid numerical issues
                val = num / den
                fsc.append(np.clip(val, -1.0, 1.0))
                freqs.append((bins[i] + bins[i + 1]) / 2.0)

    return np.array(freqs, dtype=np.float64), np.array(fsc, dtype=np.float64)


def compute_ccc(data1: npt.NDArray[Any], data2: npt.NDArray[Any]) -> float:
    """
    Compute the Cross-Correlation Coefficient (CCC) between two 3D maps.
    """
    assert data1.shape == data2.shape

    # Flatten and convert to float64 for precision
    d1 = data1.astype(np.float64).ravel()
    d2 = data2.astype(np.float64).ravel()

    d1 = d1 - np.mean(d1)
    d2 = d2 - np.mean(d2)

    num = np.sum(d1 * d2)
    den = np.sqrt(np.sum(d1**2) * np.sum(d2**2))

    if den == 0:
        return 0.0
    return float(np.clip(num / den, -1.0, 1.0))


def save_mrc(
    data: npt.NDArray[Any],
    output_path: str,
    origin: npt.ArrayLike = (0, 0, 0),
    spacing: npt.ArrayLike = (1, 1, 1),
) -> None:
    """
    Save numpy array to MRC file.
    """
    # Convert origin and spacing to 1D arrays for easier indexing if they aren't already
    origin_arr = np.asarray(origin)
    spacing_arr = np.asarray(spacing)

    with mrcfile.new(output_path, overwrite=True) as mrc:
        mrc.set_data(data.astype(np.float32))
        mrc.voxel_size = (float(spacing_arr[0]), float(spacing_arr[1]), float(spacing_arr[2]))
        # mrcfile uses x, y, z for origin
        mrc.header.origin.x = float(origin_arr[0])
        mrc.header.origin.y = float(origin_arr[1])
        mrc.header.origin.z = float(origin_arr[2])
        mrc.update_header_from_data()
